"""Dispatches characterization jobs and manages cell data"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from charlib.characterizer import utils, plots
from charlib.characterizer.cell import Cell, CellTestConfig
from charlib.characterizer.units import UnitsSettings
from charlib.characterizer.procedures import registered_procedures, ProcedureFailedException
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

class Characterizer:
    """Main object of Charlib. Keeps track of settings and cells, and schedules simulations."""

    def __init__(self, **kwargs) -> None:
        self.settings = CharacterizationSettings(**kwargs)
        self.library = Library(kwargs.pop('lib_name'), **self.settings.liberty_attrs_as_dict())
        self.cells = []

    def add_cell(self, name: str, properties: dict):
        """Add a cell to be characterized"""
        # Get pg_pins from library settings, then construct the cell
        supply_pins = {self.settings.primary_power.name: 'primary_power',
                       self.settings.primary_ground.name: 'primary_ground',
                       self.settings.pwell.name: 'pwell',
                       self.settings.nwell.name: 'nwell'}
        try:
            cell = Cell(name, supply_pins, **properties)
        except Exception as e: # FIXME: We should have a more specific error type than this!
            if self.settings.omit_on_failure:
                return
            else:
                raise ValueError(f'Unable to add cell {name}') from e

        # Handle keywords for plots
        if properties.get('plots', []) == 'all':
            properties['plots'] = ['delay', 'io']
        config = CellTestConfig(properties.pop('models'), **properties)
        self.cells.append((cell, config))

    def analyse_cell(self, cell, config) -> list:
        """Return a list of callable characterization tasks required for this cell."""
        simulations = []

        # Measure input pin capacitances
        simulations += self.settings.simulation.input_capacitance(cell, config, self.settings)

        # Identify which delay and constraint procedures to run based on cell & config
        if cell.is_sequential:
            # Find setup & hold constraints (clock-to-q, en-to-q)
            simulations += self.settings.simulation.metastability_constraint(cell, config, self.settings)
            # TODO: Find minimum pulse width constraints (set, reset, enable, clock)
            # Find recovery & removal constraints (clk/en-to-set, clk/en-to-reset)
            simulations += self.settings.simulation.recovery_constraint(cell, config, self.settings)
            simulations += self.settings.simulation.removal_constraint(cell, config, self.settings)
            # Measure sequential propagation and transient delays
            simulations += self.settings.simulation.sequential_delay(cell, config, self.settings)
        else:
            # Measure combinational propagation and transient delays
            simulations += self.settings.simulation.combinational_delay(cell, config, self.settings)
            # Measure static leakage power for all input states
            simulations += self.settings.simulation.combinational_leakage(cell, config, self.settings)
        return simulations

    def characterize(self):
        """Execute scheduled simulation jobs in parallel"""
        engine = getattr(self.settings, 'execution_engine', 'legacy')
        if engine == 'reusable_ngspice':
            return self._characterize_reusable_ngspice()
        # Setup: Prepare simulation jobs single-threadedly (is that a word?)
        simulation_tasks = []
        for (cell, config) in self.cells:
            simulation_tasks += self.analyse_cell(cell, config)

        # Run all simulation jobs and merge each resulting liberty cell group into the library
        with tqdm(bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                  total=len(simulation_tasks), desc="Characterizing") as progress_bar:
            with ProcessPoolExecutor(max_workers=self.settings.jobs) as executor:
                futures = [executor.submit(task, *args) for (task, *args) in simulation_tasks]
                for future in as_completed(futures):
                    try:
                        cell_group = future.result()
                    except ProcedureFailedException:
                        if self.settings.omit_on_failure:
                            continue
                        else:
                            raise
                    self.library.add_group(cell_group)
                    progress_bar.update(1)

        # Post-processing: Fetch generated table templates and add them to the library
        lut_templates = []
        for timing_group in self.library.subgroups_with_name('timing'):
            lut_templates += [lut_group.template for lut_group in timing_group.groups.values()]
        [self.library.add_group(lut_template) for lut_template in lut_templates]

        # Plot delay surfaces (if desired)
        for (cell, config) in self.cells:
            cell_group = self.library.group('cell', cell.name)
            if 'delay' in config.plots:
                for pin_group in cell_group.subgroups_with_name('pin'):
                    pin = pin_group.identifier
                    for timing_group in pin_group.subgroups_with_name('timing'):
                        related_pin = timing_group.attributes['related_pin'].value
                        fig = plots.plot_delay_surfaces(list(timing_group.groups.values()),
                                                        title=f'Cell delays ({related_pin} to {pin})')
                        # FIXME: let user decide whether to show or save
                        fig_path = self.settings.plots_dir / cell.name
                        fig_path.mkdir(parents=True, exist_ok=True)
                        fig.savefig(fig_path / f'{related_pin} to {pin} delay.png') # FIXME: filetype should be configurable
                        plt.close()
        return self.library.to_liberty(precision=6)

    def _characterize_reusable_ngspice(self):
        """Execute combinational delay characterization using reusable worker contexts.

        Plans all combinational delay points, groups them by topology signature so that
        each signature loads its deck exactly once, packs signatures into batches using
        LPT bin-packing over point counts, dispatches WorkerBatchRequests objects to a
        process pool with per-process WorkerContexts, then deterministically assembles
        the resulting Liberty groups.
        """
        import hashlib
        from collections import defaultdict
        from charlib.characterizer.reusable_engine import (
            WorkerContext, WorkerRequest, WorkerResult, execute_request, LibertyAssembler,
            TopologySignature, MeasurementResult, detect_missing_requests, init_worker,
            SignatureGroup, WorkerBatchRequest, WorkerBatchResult, execute_batch,
            lpt_batch_groups,
        )
        from charlib.characterizer.procedures.combinational.delay import (
            plan_points, _build_deck, reduce_condition_results, assemble_delay_liberty,
        )

        os.environ.setdefault('OMP_NUM_THREADS', '1')

        # Determine the reduction criterion from the configured combinational delay procedure
        if self.settings.simulation.combinational_delay.__name__ == 'combinational_average':
            from numpy import average as criterion_func
        else:
            criterion_func = max

        work_items = []
        group_info = {}
        next_group_id = 0
        cells_to_process = []

        for cell, config in self.cells:
            if cell.is_sequential:
                # Reusable worker contexts are currently limited to combinational cells.
                continue
            cells_to_process.append((cell, config))
            for variation in config.variations('data_slews', 'loads', 'transient_sim_end_time'):
                for path in cell.paths():
                    group_id = next_group_id
                    next_group_id += 1
                    group_info[group_id] = (cell, config, variation, path, criterion_func)

                    data_slew = variation['data_slews'] * self.settings.units.time
                    load = variation['loads'] * self.settings.units.capacitance
                    t_sim_end = max(variation['transient_sim_end_time'] * self.settings.units.time,
                                    1000 * data_slew)
                    vdd = self.settings.primary_power.voltage * self.settings.units.voltage
                    vss = self.settings.primary_ground.voltage * self.settings.units.voltage

                    for point in plan_points(cell, config, self.settings, variation, path, criterion_func):
                        state_map = dict(point.state_condition)
                        deck_text, pin_map, measurement_names, stable_pins_map_str = _build_deck(
                            cell, config, self.settings, variation, path, state_map,
                            data_slew, load, t_sim_end, vdd, vss)
                        signature = TopologySignature(
                            cell_hash=hashlib.md5(point.cell_name.encode()).hexdigest()[:16],
                            netlist_hash=hashlib.md5(point.netlist_path.encode()).hexdigest()[:16],
                            model_hashes=tuple((s or '', hashlib.md5(p.encode()).hexdigest()[:16])
                                              for s, p in point.model_paths),
                            pin_topology=tuple((p.name, p.role.name,
                                                pin_map.target_inputs.get(p.name,
                                                    pin_map.target_outputs.get(p.name,
                                                        pin_map.stable_inputs.get(p.name, ''))))
                                               for p in cell.pins_in_netlist_order()),
                            state_condition=point.state_condition,
                            measurement_names=tuple(sorted(measurement_names)),
                            measurement_directions=tuple(),
                            backend=self.settings.simulation.backend,
                            temperature=point.temperature,
                            supplies_hash=hashlib.md5(str(point.supplies).encode()).hexdigest()[:16],
                            data_slew=float(data_slew),
                            t_sim_end=float(t_sim_end),
                        )
                        work_items.append((point, group_id, signature, deck_text, measurement_names))

        # Group points by signature load key. All points that share a signature share a deck.
        signature_groups: dict = {}
        point_to_group_id = {}
        expected_point_ids = []
        for point, group_id, signature, deck_text, measurement_names in work_items:
            expected_point_ids.append(point.point_id)
            point_to_group_id[point.point_id] = group_id
            key = WorkerContext._signature_load_key(signature)
            if key not in signature_groups:
                signature_groups[key] = SignatureGroup(
                    signature=signature,
                    deck_text=deck_text,
                    measurement_names=tuple(sorted(measurement_names)),
                    sweep_points=(),
                )
                # Defensive: ensure deck_text/measurement_names are consistent for the key.
                # If two points produce the same signature but different decks, keep the first.
            existing = signature_groups[key]
            existing = SignatureGroup(
                signature=existing.signature,
                deck_text=existing.deck_text,
                measurement_names=existing.measurement_names,
                sweep_points=existing.sweep_points + (point,),
            )
            signature_groups[key] = existing

        # LPT sort by point count descending, then pack into 2N..4N micro-batches.
        num_workers = self.settings.jobs if self.settings.jobs is not None else 1
        batches = lpt_batch_groups(list(signature_groups.values()), num_workers)

        batch_requests = []
        for index, batch_groups in enumerate(batches):
            if not batch_groups:
                continue
            batch_requests.append(WorkerBatchRequest(
                batch_id=f"batch_{index}",
                groups=tuple(batch_groups),
            ))

        # Convert old per-point request objects only for missing-point detection.
        requests = [
            WorkerRequest(request_id=point.point_id, point=point, deck_text='', signature=signature)
            for point, _, signature, _, _ in work_items
        ]
        request_map = {req.request_id: req for req in requests}

        results_by_id = {}
        if batch_requests:
            max_workers = num_workers
            with ProcessPoolExecutor(max_workers=max_workers,
                                     initializer=init_worker,
                                     initargs=(0,)) as executor:
                futures = {executor.submit(execute_batch, request): request.batch_id
                           for request in batch_requests}
                for future in as_completed(futures):
                    request_id = futures[future]
                    try:
                        batch_result = future.result()
                        for worker_result in batch_result.results:
                            results_by_id[worker_result.request_id] = worker_result.result
                    except Exception:
                        # Entire batch crashed; results will be reported via missing detection.
                        pass

        # Detect any missing results and record them without producing partial Liberty.
        missing_ids = detect_missing_requests(
            requests,
            [WorkerResult(request_id=point_id, result=result)
             for point_id, result in results_by_id.items()])
        if missing_ids:
            for request_id in missing_ids:
                task_id = request_map[request_id].point.task_id
                results_by_id[request_id] = MeasurementResult(
                    point_id=request_id,
                    task_id=task_id,
                    status='error',
                    error='Result missing after batch execution',
                )
            raise RuntimeError(
                f"Reusable ngspice characterization failed; {len(missing_ids)} result(s) missing: "
                f"{missing_ids[:10]}"
            )

        # Fail early if any individual point returned an error status.
        error_ids = [point_id for point_id, result in results_by_id.items()
                     if result.status != 'ok']
        if error_ids:
            raise RuntimeError(
                f"Reusable ngspice characterization failed for {len(error_ids)} point(s): "
                f"{error_ids[:10]}"
            )

        # Deterministic final assembly ordered by point_id.
        assembler = LibertyAssembler()
        assembler.add_results(list(results_by_id.values()))
        ordered_results = assembler.ordered_results()

        # Map each ordered result back to its original group_id for assembly.
        groups_map = defaultdict(list)
        for result in ordered_results:
            group_id = point_to_group_id.get(result.point_id)
            if group_id is not None:
                groups_map[group_id].append(result)

        for group_id, results in groups_map.items():
            cell, config, variation, path, criterion_func = group_info[group_id]
            reduced = reduce_condition_results(results, criterion_func)
            assemble_delay_liberty(cell, config, self.settings, variation, path, reduced)

        for cell, config in cells_to_process:
            self.library.add_group(cell.liberty)

        # Legacy fallback for non-delay procedures
        legacy_tasks = []
        for cell, config in self.cells:
            if cell.is_sequential:
                # Sequential: all procedures via legacy
                tasks = self.analyse_cell(cell, config)
            else:
                # Combinational: only non-delay procedures via legacy
                tasks = [t for t in self.analyse_cell(cell, config)
                         if t[0].__name__ not in ('combinational_worst_case', 'combinational_average')]
            legacy_tasks.extend(tasks)

        if legacy_tasks:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from charlib.characterizer.procedures import ProcedureFailedException
            with ProcessPoolExecutor(max_workers=self.settings.jobs) as executor:
                futures = {executor.submit(t[0], *t[1:]): i for i, t in enumerate(legacy_tasks)}
                for future in as_completed(futures):
                    try:
                        cell_group = future.result()
                        self.library.add_group(cell_group)
                    except ProcedureFailedException:
                        if self.settings.omit_on_failure:
                            continue
                        raise

        # Post-processing: Fetch generated table templates and add them to the library
        lut_templates = []
        for timing_group in self.library.subgroups_with_name('timing'):
            lut_templates += [lut_group.template for lut_group in timing_group.groups.values()]
        [self.library.add_group(lut_template) for lut_template in lut_templates]

        # Plot delay surfaces (if desired)
        for (cell, config) in self.cells:
            cell_group = self.library.group('cell', cell.name)
            if 'delay' in config.plots:
                for pin_group in cell_group.subgroups_with_name('pin'):
                    pin = pin_group.identifier
                    for timing_group in pin_group.subgroups_with_name('timing'):
                        related_pin = timing_group.attributes['related_pin'].value
                        fig = plots.plot_delay_surfaces(list(timing_group.groups.values()),
                                                        title=f'Cell delays ({related_pin} to {pin})')
                        # FIXME: let user decide whether to show or save
                        fig_path = self.settings.plots_dir / cell.name
                        fig_path.mkdir(parents=True, exist_ok=True)
                        fig.savefig(fig_path / f'{related_pin} to {pin} delay.png')
                        plt.close()


        return self.library.to_liberty(precision=6)
class CharacterizationSettings:
    """Container for characterization settings"""
    def __init__(self, **kwargs):
        """Create a new CharacterizationSettings instance"""
        # Behavioral settings
        if 'jobs' in kwargs:
            self.jobs = kwargs.pop('jobs')
        else:
            self.jobs = None if kwargs.pop('multithreaded', True) else 1
        self.results_dir = Path(kwargs.pop('results_dir', 'results'))
        self.plots_dir = self.results_dir / 'plots'
        self.debug = kwargs.pop('debug', False)
        self.debug_dir = Path(kwargs.pop('debug_dir', 'debug'))
        self.quiet = kwargs.pop('quiet', False)
        self.cell_defaults = kwargs.get('cell_defaults', {})
        self.omit_on_failure = kwargs.get('omit_on_failure', False)
        self.execution_engine = kwargs.get('execution_engine',
                                           kwargs.get('simulation', {}).get('execution_engine', 'legacy'))

        # Simulation procedures
        self.simulation = SimulationSettings(**kwargs.get('simulation', {}))

        # Units for simulation and results
        self.units = UnitsSettings(**kwargs.get('units', {}))

        # Library-wide named voltages
        nodes = kwargs.pop('named_nodes', {})
        self.primary_power = NamedNode(**nodes.get('primary_power', {'name':'VDD', 'voltage': 3.3}))
        self.primary_ground = NamedNode(**nodes.get('primary_ground', {'name':'VSS', 'voltage': 0}))
        self.pwell = NamedNode(**nodes.get('pwell', {'name':'VPW', 'voltage': 0}))
        self.nwell = NamedNode(**nodes.get('nwell', {'name':'VNW', 'voltage': 3.3}))

        # Logic thresholds
        self.logic_thresholds = LogicThresholds(**kwargs.get('logic_thresholds', {}))

        # Operating conditions
        self.temperature = kwargs.get('temperature', 25)

    @property
    def named_nodes(self):
        """Convenience accessor returning a tuple of all named nodes"""
        return (self.primary_power, self.primary_ground, self.nwell, self.pwell)

    def liberty_attrs_as_dict(self):
        """Return a dict of library-wide settings that should be written to the liberty file."""
        spice_unit = lambda unit: f'1{unit.prefixed_unit.str_spice()}'
        return {
            'nom_voltage': self.primary_power.voltage,
            'nom_temperature': self.temperature,
            'time_unit': spice_unit(self.units.time),
            'voltage_unit': spice_unit(self.units.voltage),
            'current_unit': spice_unit(self.units.current),
            'pulling_resistance_unit': spice_unit(self.units.current),
            'leakage_power_unit': spice_unit(self.units.power),
            'capacitive_load_unit': [1, self.units.capacitance.prefixed_unit.str_spice()],
            'slew_upper_threshold_pct_rise': self.logic_thresholds.high,
            'slew_lower_threshold_pct_rise': self.logic_thresholds.low,
            'slew_upper_threshold_pct_fall': self.logic_thresholds.high,
            'slew_lower_threshold_pct_fall': self.logic_thresholds.low,
            'input_threshold_pct_rise': self.logic_thresholds.rising,
            'input_threshold_pct_fall': self.logic_thresholds.falling,
            'output_threshold_pct_rise': self.logic_thresholds.rising,
            'output_threshold_pct_rise': self.logic_thresholds.rising,
            'output_threshold_pct_fall': self.logic_thresholds.falling,
        }

class SimulationSettings:
    """Container for simulation backend and procedures"""
    def __init__(self, **kwargs):
        self.backend = kwargs.get('backend', 'ngspice-shared')
        self.execution_engine = kwargs.get('execution_engine', 'legacy')
        self.input_capacitance = registered_procedures[
            kwargs.get('input_capacitance_procedure', 'ac_sweep')
        ]['callable']
        self.combinational_delay = registered_procedures[
            kwargs.get('combinational_delay_procedure', 'combinational_worst_case')
        ]['callable']
        self.combinational_leakage = registered_procedures[
            kwargs.get('combinational_leakage_procedure', 'combinational_leakage')
        ]['callable']
        self.sequential_delay = registered_procedures[
            kwargs.get('sequential_delay_procedure', 'sequential_worst_case')
        ]['callable']
        self.metastability_constraint = registered_procedures[
            kwargs.get('setup_hold_constraint_procedure', 'measure_setup_hold_from_contour')
        ]['callable']
        self.recovery_constraint = registered_procedures[
            kwargs.get('recovery_constraint_procedure', 'recovery_constraint')
        ]['callable']
        self.removal_constraint = registered_procedures[
            kwargs.get('removal_constraint_procedure', 'removal_constraint')
        ]['callable']
        self.min_pulse_width_constraint = registered_procedures[
            kwargs.get('min_pulse_width_constraint_procedure', 'min_pulse_width_constraint')
        ]['callable']

class LogicThresholds:
    """Container for logic_thresholds settings"""
    def __init__(self, **kwargs):
        self.low = kwargs.get('low', 0.2)
        self.high = kwargs.get('high', 0.8)
        self.rising = kwargs.get('rising', 0.5)
        self.falling = kwargs.get('falling', 0.5)

class NamedNode:
    """Binds supply node names to voltages"""
    def __init__(self, name, voltage = 0):
        self.name = name
        self.voltage = voltage

    def __str__(self) -> str:
        return f'Name: {self.name}\nVoltage: {self.voltage}'

    def __repr__(self) -> str:
        return f'NamedNode({self.name}, {self.voltage})'

    @property
    def subscript(self) -> str:
        """Return the 'subscript' portion of the voltage name e.g. Vdd -> dd"""
        return self.name[1:] if self.name.lower().startswith('v') else self.name
