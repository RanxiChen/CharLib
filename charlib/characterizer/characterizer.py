"""Dispatches characterization jobs and manages cell data — Phase C: batch ProcessPool"""

import time
import pickle
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from charlib.characterizer import utils, plots
from charlib.characterizer.cell import Cell, CellTestConfig
from charlib.characterizer.units import UnitsSettings
from charlib.characterizer.procedures import registered_procedures, ProcedureFailedException
from charlib.characterizer.procedures.combinational.delay import (
    batch_run_compact_delay_points,
    build_liberty_from_compact_rows,
)
from charlib.liberty.library import Library

import charlib.characterizer.procedures.pin_capacitance.ac_sweep
import charlib.characterizer.procedures.combinational.delay
import charlib.characterizer.procedures.combinational.leakage_power
import charlib.characterizer.procedures.sequential.delay
import charlib.characterizer.procedures.sequential.constraint.metastability.binary_search
import charlib.characterizer.procedures.sequential.constraint.metastability.c2q_contour
import charlib.characterizer.procedures.sequential.constraint.recovery
import charlib.characterizer.procedures.sequential.constraint.removal
import charlib.characterizer.procedures.sequential.constraint.min_pulse_width

# ── Pickle instrumentation ────────────────────────────────────────────────
_ORIG_DUMPS = pickle.dumps
_ORIG_LOADS = pickle.loads
_PICKLE_DUMP_TIME = 0.0
_PICKLE_LOAD_TIME = 0.0
_PICKLE_DUMP_BYTES = 0
_PICKLE_LOAD_BYTES = 0

def _instrumented_dumps(obj, *args, **kwargs):
    global _PICKLE_DUMP_TIME, _PICKLE_DUMP_BYTES
    t0 = time.perf_counter()
    result = _ORIG_DUMPS(obj, *args, **kwargs)
    _PICKLE_DUMP_TIME += time.perf_counter() - t0
    _PICKLE_DUMP_BYTES += len(result)
    return result

def _instrumented_loads(data, *args, **kwargs):
    global _PICKLE_LOAD_TIME, _PICKLE_LOAD_BYTES
    t0 = time.perf_counter()
    result = _ORIG_LOADS(data, *args, **kwargs)
    _PICKLE_LOAD_TIME += time.perf_counter() - t0
    _PICKLE_LOAD_BYTES += len(data)
    return result

def _install_pickle_hooks():
    pickle.dumps = _instrumented_dumps
    pickle.loads = _instrumented_loads

def _remove_pickle_hooks():
    pickle.dumps = _ORIG_DUMPS
    pickle.loads = _ORIG_LOADS

def _reset_pickle_counters():
    global _PICKLE_DUMP_TIME, _PICKLE_LOAD_TIME, _PICKLE_DUMP_BYTES, _PICKLE_LOAD_BYTES
    _PICKLE_DUMP_TIME = 0.0
    _PICKLE_LOAD_TIME = 0.0
    _PICKLE_DUMP_BYTES = 0
    _PICKLE_LOAD_BYTES = 0


def _is_delay_task(task_tuple):
    """Check if a task tuple is a combinational delay measurement task."""
    if not task_tuple or len(task_tuple) < 1:
        return False
    task_fn = task_tuple[0]
    fn_name = getattr(task_fn, "__name__", "")
    return "measure_delays_for_path" in fn_name


class Characterizer:
    """Main object of Charlib. Keeps track of settings and cells, and schedules simulations."""

    def __init__(self, **kwargs) -> None:
        self.settings = CharacterizationSettings(**kwargs)
        self.library = Library(kwargs.pop("lib_name"), **self.settings.liberty_attrs_as_dict())
        self.cells = []
        self.metrics = {}

    def add_cell(self, name: str, properties: dict):
        supply_pins = {self.settings.primary_power.name: "primary_power",
                       self.settings.primary_ground.name: "primary_ground",
                       self.settings.pwell.name: "pwell",
                       self.settings.nwell.name: "nwell"}
        try:
            cell = Cell(name, supply_pins, **properties)
        except Exception as e:
            if self.settings.omit_on_failure:
                return
            else:
                raise ValueError(f"Unable to add cell {name}") from e
        if properties.get("plots", []) == "all":
            properties["plots"] = ["delay", "io"]
        config = CellTestConfig(properties.pop("models"), **properties)
        self.cells.append((cell, config))

    def analyse_cell(self, cell, config) -> list:
        simulations = []
        simulations += self.settings.simulation.input_capacitance(cell, config, self.settings)
        if cell.is_sequential:
            simulations += self.settings.simulation.metastability_constraint(cell, config, self.settings)
            simulations += self.settings.simulation.recovery_constraint(cell, config, self.settings)
            simulations += self.settings.simulation.removal_constraint(cell, config, self.settings)
            simulations += self.settings.simulation.sequential_delay(cell, config, self.settings)
        else:
            simulations += self.settings.simulation.combinational_delay(cell, config, self.settings)
            simulations += self.settings.simulation.combinational_leakage(cell, config, self.settings)
        return simulations

    def _batch_delay_tasks(self, cell, config, delay_tasks):
        batch_points = []
        for (task_fn, cell_arg, config_arg, settings_arg, variation, path, criterion) in delay_tasks:
            for state_map in cell.nonmasking_conditions_for_path(*path):
                batch_points.append((variation, path, state_map))

        if not batch_points:
            return [], 0

        jobs = max(1, self.settings.jobs or 1)
        target_batch_count = max(jobs, min(2 * jobs, len(batch_points)))
        batch_size = max(1, ceil(len(batch_points) / target_batch_count))

        batches = []
        for i in range(0, len(batch_points), batch_size):
            batches.append(batch_points[i:i + batch_size])

        self.metrics["delay_point_count"] = len(batch_points)
        self.metrics["delay_batch_count"] = len(batches)
        self.metrics["delay_batch_size"] = batch_size

        batch_specs = []
        for batch in batches:
            batch_specs.append((batch_run_compact_delay_points, cell, config, self.settings, batch))
        return batch_specs, len(batch_points)

    def characterize(self):
        m = self.metrics
        _reset_pickle_counters()

        # ── Task generation ──
        t0 = time.perf_counter()
        all_tasks = []
        for (cell_arg, config_arg) in self.cells:
            all_tasks.append((cell_arg, config_arg, self.analyse_cell(cell_arg, config_arg)))
        m["task_gen_s"] = time.perf_counter() - t0

        # Separate and batch
        batched_specs = []
        non_delay_tasks = []
        total_delay_points = 0

        for (cell_arg, config_arg, tasks) in all_tasks:
            delay_tasks = [t for t in tasks if _is_delay_task(t)]
            other_tasks = [t for t in tasks if not _is_delay_task(t)]
            non_delay_tasks.extend(other_tasks)
            if delay_tasks:
                specs, n_points = self._batch_delay_tasks(cell_arg, config_arg, delay_tasks)
                batched_specs.extend(specs)
                total_delay_points += n_points

        simulation_tasks = batched_specs + non_delay_tasks
        m["task_count"] = len(simulation_tasks)
        m["actual_simulator_run_task_count"] = total_delay_points + len(non_delay_tasks)

        # ── Pool create ──
        t0 = time.perf_counter()
        executor = ProcessPoolExecutor(max_workers=self.settings.jobs)
        m["pool_create_s"] = time.perf_counter() - t0

        # ── Submit loop ──
        _install_pickle_hooks()
        t0 = time.perf_counter()
        futures = [executor.submit(task, *args) for (task, *args) in simulation_tasks]
        m["submit_s"] = time.perf_counter() - t0
        m["submit_count"] = len(futures)

        # ── Result receive + merge ──
        recv_time = 0.0
        merge_time = 0.0
        compact_rows_by_cell = {}

        with tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  total=len(simulation_tasks), desc="Characterizing") as progress_bar:
            for future in as_completed(futures):
                t_recv = time.perf_counter()
                try:
                    result = future.result()
                except ProcedureFailedException:
                    if self.settings.omit_on_failure:
                        recv_time += time.perf_counter() - t_recv
                        progress_bar.update(1)
                        continue
                    else:
                        raise
                recv_time += time.perf_counter() - t_recv

                if isinstance(result, list):
                    for row in result:
                        op = row["output_pin"]
                        if op not in compact_rows_by_cell:
                            compact_rows_by_cell[op] = []
                        compact_rows_by_cell[op].append(row)
                else:
                    t_merge = time.perf_counter()
                    self.library.add_group(result)
                    merge_time += time.perf_counter() - t_merge

                progress_bar.update(1)

        m["result_recv_s"] = recv_time

        # Build Liberty from compact rows
        t_merge = time.perf_counter()
        for (cell_arg, config_arg) in self.cells:
            cell_rows = []
            for op in compact_rows_by_cell:
                cell_rows.extend(compact_rows_by_cell[op])
            if cell_rows:
                cell_group = build_liberty_from_compact_rows(cell_arg, config_arg, self.settings, cell_rows, max)
                self.library.add_group(cell_group)
        merge_time += time.perf_counter() - t_merge
        m["liberty_merge_s"] = merge_time

        m["pickle_dump_s"] = _PICKLE_DUMP_TIME
        m["pickle_load_s"] = _PICKLE_LOAD_TIME
        m["pickle_dump_bytes"] = _PICKLE_DUMP_BYTES
        m["pickle_load_bytes"] = _PICKLE_LOAD_BYTES
        _remove_pickle_hooks()

        # ── Pool shutdown ──
        t0 = time.perf_counter()
        executor.shutdown(wait=True)
        m["pool_shutdown_s"] = time.perf_counter() - t0

        # Post-processing
        lut_templates = []
        for timing_group in self.library.subgroups_with_name("timing"):
            lut_templates += [lut_group.template for lut_group in timing_group.groups.values()]
        [self.library.add_group(lut_template) for lut_template in lut_templates]

        # Plot delay surfaces
        for (cell, config) in self.cells:
            cell_group = self.library.group("cell", cell.name)
            if "delay" in config.plots:
                for pin_group in cell_group.subgroups_with_name("pin"):
                    pin = pin_group.identifier
                    for timing_group in pin_group.subgroups_with_name("timing"):
                        related_pin = timing_group.attributes["related_pin"].value
                        fig = plots.plot_delay_surfaces(list(timing_group.groups.values()),
                                                        title=f"Cell delays ({related_pin} to {pin})")
                        fig_path = self.settings.plots_dir / cell.name
                        fig_path.mkdir(parents=True, exist_ok=True)
                        fig.savefig(fig_path / f"{related_pin} to {pin} delay.png")
                        import matplotlib.pyplot as plt
                        plt.close()
        return self.library.to_liberty(precision=6)


class CharacterizationSettings:
    def __init__(self, **kwargs):
        self.jobs = None if kwargs.pop("multithreaded", True) else 1
        self.results_dir = Path(kwargs.pop("results_dir", "results"))
        self.plots_dir = self.results_dir / "plots"
        self.debug = kwargs.pop("debug", False)
        self.debug_dir = Path(kwargs.pop("debug_dir", "debug"))
        self.quiet = kwargs.pop("quiet", False)
        self.cell_defaults = kwargs.get("cell_defaults", {})
        self.omit_on_failure = kwargs.get("omit_on_failure", False)
        self.simulation = SimulationSettings(**kwargs.get("simulation", {}))
        self.units = UnitsSettings(**kwargs.get("units", {}))
        nodes = kwargs.pop("named_nodes", {})
        self.primary_power = NamedNode(**nodes.get("primary_power", {"name":"VDD", "voltage": 3.3}))
        self.primary_ground = NamedNode(**nodes.get("primary_ground", {"name":"VSS", "voltage": 0}))
        self.pwell = NamedNode(**nodes.get("pwell", {"name":"VPW", "voltage": 0}))
        self.nwell = NamedNode(**nodes.get("nwell", {"name":"VNW", "voltage": 3.3}))
        self.logic_thresholds = LogicThresholds(**kwargs.get("logic_thresholds", {}))
        self.temperature = kwargs.get("temperature", 25)

    @property
    def named_nodes(self):
        return (self.primary_power, self.primary_ground, self.nwell, self.pwell)

    def liberty_attrs_as_dict(self):
        spice_unit = lambda unit: f"1{unit.prefixed_unit.str_spice()}"
        return {
            "nom_voltage": self.primary_power.voltage,
            "nom_temperature": self.temperature,
            "time_unit": spice_unit(self.units.time),
            "voltage_unit": spice_unit(self.units.voltage),
            "current_unit": spice_unit(self.units.current),
            "pulling_resistance_unit": spice_unit(self.units.current),
            "leakage_power_unit": spice_unit(self.units.power),
            "capacitive_load_unit": [1, self.units.capacitance.prefixed_unit.str_spice()],
            "slew_upper_threshold_pct_rise": self.logic_thresholds.high,
            "slew_lower_threshold_pct_rise": self.logic_thresholds.low,
            "slew_upper_threshold_pct_fall": self.logic_thresholds.high,
            "slew_lower_threshold_pct_fall": self.logic_thresholds.low,
            "input_threshold_pct_rise": self.logic_thresholds.rising,
            "input_threshold_pct_fall": self.logic_thresholds.falling,
            "output_threshold_pct_rise": self.logic_thresholds.rising,
            "output_threshold_pct_fall": self.logic_thresholds.falling,
        }

class SimulationSettings:
    def __init__(self, **kwargs):
        self.backend = kwargs.get("backend", "ngspice-shared")
        self.input_capacitance = registered_procedures[
            kwargs.get("input_capacitance_procedure", "ac_sweep")
        ]["callable"]
        self.combinational_delay = registered_procedures[
            kwargs.get("combinational_delay_procedure", "combinational_worst_case")
        ]["callable"]
        self.combinational_leakage = registered_procedures[
            kwargs.get("combinational_leakage_procedure", "combinational_leakage")
        ]["callable"]
        self.sequential_delay = registered_procedures[
            kwargs.get("sequential_delay_procedure", "sequential_worst_case")
        ]["callable"]
        self.metastability_constraint = registered_procedures[
            kwargs.get("setup_hold_constraint_procedure", "measure_setup_hold_from_contour")
        ]["callable"]
        self.recovery_constraint = registered_procedures[
            kwargs.get("recovery_constraint_procedure", "recovery_constraint")
        ]["callable"]
        self.removal_constraint = registered_procedures[
            kwargs.get("removal_constraint_procedure", "removal_constraint")
        ]["callable"]
        self.min_pulse_width_constraint = registered_procedures[
            kwargs.get("min_pulse_width_constraint_procedure", "min_pulse_width_constraint")
        ]["callable"]

class LogicThresholds:
    def __init__(self, **kwargs):
        self.low = kwargs.get("low", 0.2)
        self.high = kwargs.get("high", 0.8)
        self.rising = kwargs.get("rising", 0.5)
        self.falling = kwargs.get("falling", 0.5)

class NamedNode:
    def __init__(self, name, voltage = 0):
        self.name = name
        self.voltage = voltage
    def __str__(self) -> str:
        return f"Name: {self.name}\nVoltage: {self.voltage}"
    def __repr__(self) -> str:
        return f"NamedNode({self.name}, {self.voltage})"
    @property
    def subscript(self) -> str:
        return self.name[1:] if self.name.lower().startswith("v") else self.name
