"""Dispatches characterization jobs and manages cell data"""

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import signal
import json

from charlib.characterizer import utils, plots
from charlib.characterizer.cell import Cell, CellTestConfig
from charlib.characterizer.units import UnitsSettings
from charlib.characterizer.procedures import registered_procedures, ProcedureFailedException
from charlib.liberty.library import Library
from charlib.characterizer.scheduler import TaskRecord, TaskResult, SchedulerMetrics, BatchRecord, BatchResult, plan_batches, execute_batch

import charlib.characterizer.procedures.pin_capacitance.ac_sweep
import charlib.characterizer.procedures.pin_capacitance.charge_integration
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

    def _unfinished_task_ids(self, records, all_results):
        """Return sorted list of task IDs in records that are not present in all_results."""
        finished = {r.task_id for r in all_results}
        return sorted(record.task_id for record in records if record.task_id not in finished)

    def _apply_batched_results(self, all_results, records, omit_on_failure):
        """Deterministic merge with duplicate/missing ID checks and omit_on_failure handling."""
        all_results.sort(key=lambda r: r.task_id)
        ids = [r.task_id for r in all_results]
        if len(ids) != len(set(ids)):
            from collections import Counter
            dupes = [tid for tid, count in Counter(ids).items() if count > 1]
            raise RuntimeError(f"Duplicate task IDs in batch results: {dupes}")
        expected = set(range(len(records)))
        actual = set(ids)
        if expected != actual:
            missing = expected - actual
            extra = actual - expected
            raise RuntimeError(
                f"Task ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

        if omit_on_failure:
            # Apply only successful results
            for r in all_results:
                if r.error_type is None and r.value is not None:
                    self.library.add_group(r.value)
        else:
            # Fail on first error
            for r in all_results:
                if r.error_type is not None:
                    if r.exception is not None:
                        raise r.exception
                    raise RuntimeError(
                        f"Task {r.task_id} failed: {r.error_type}: {r.error_message}"
                    )
            # All successful — apply all
            for r in all_results:
                self.library.add_group(r.value)

    def _characterize_batched(self, records, progress_bar, metrics):
        """Execute TaskRecords using cost-ordered micro-batches.

        Activated when ``settings.simulation.execution_engine`` is ``batched``.
        """
        capture = self.settings.scheduler_metrics_path is not None
        metrics.capture_metrics = capture

        workers = self.settings.jobs or os.cpu_count() or 4
        batch_factor = self.settings.simulation.scheduler_batch_factor
        max_inflight_per_worker = self.settings.simulation.scheduler_max_inflight_per_worker

        metrics.requested_jobs = self.settings.jobs
        metrics.resolved_max_workers = workers
        metrics.max_in_flight = max_inflight_per_worker * workers

        submit_start = time.perf_counter()
        batches = plan_batches(records, workers, batch_factor=batch_factor)
        metrics.batch_count = len(batches)
        metrics.future_count = len(batches)

        future_to_batch = {}
        pending = set()
        all_results = []
        all_batch_results = []
        interrupted = False

        old_sigterm = None

        def _handle_sigterm(signum, frame):
            nonlocal interrupted
            interrupted = True
            raise KeyboardInterrupt()

        import threading
        if threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGTERM'):
            old_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)

        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
            max_in_flight = max(1, max_inflight_per_worker * workers)
            batch_iter = iter(batches)
            for _ in range(min(len(batches), max_in_flight)):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                if interrupted:
                    break
                fut = executor.submit(execute_batch, batch, capture)
                future_to_batch[fut] = batch
                pending.add(fut)

            metrics.submit_seconds = time.perf_counter() - submit_start
            collect_start = time.perf_counter()

            while pending and not interrupted:
                try:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    interrupted = True
                    break
                for fut in done:
                    batch = future_to_batch.pop(fut, None)
                    try:
                        batch_result = fut.result()
                        all_results.extend(batch_result.task_results)
                        all_batch_results.append(batch_result)
                        progress_bar.update(len(batch.tasks))
                    except Exception as e:
                        # The future itself failed (e.g., worker crash). Record an error
                        # for every task in the batch so deterministic merge can report it.
                        if batch is not None:
                            progress_bar.update(len(batch.tasks))
                            synthetic_results = [
                                TaskResult(
                                    task_id=t.task_id,
                                    error_type='ProcessPoolFailure',
                                    error_message=str(e),
                                )
                                for t in batch.tasks
                            ]
                            all_results.extend(synthetic_results)
                            all_batch_results.append(BatchResult(
                                batch_id=batch.batch_id,
                                task_results=synthetic_results,
                                worker_pid=None,
                                wall_seconds=None,
                                predicted_cost=batch.predicted_cost,
                                task_ids=[t.task_id for t in batch.tasks],
                            ))
                # Keep the pipeline full
                while len(pending) < max_in_flight and not interrupted:
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    fut = executor.submit(execute_batch, batch, capture)
                    future_to_batch[fut] = batch
                    pending.add(fut)

            metrics.collect_seconds = time.perf_counter() - collect_start
        finally:
            if executor is not None:
                if interrupted:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    # Ensure child processes are terminated
                    import multiprocessing
                    for child in multiprocessing.active_children():
                        try:
                            child.terminate()
                            child.join(timeout=1.0)
                        except Exception:
                            pass
                else:
                    executor.shutdown(wait=True, cancel_futures=True)
            if old_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, old_sigterm)
                except ValueError:
                    # Can fail if called from a non-main thread
                    pass

        if interrupted:
            unfinished = self._unfinished_task_ids(records, all_results)
            raise RuntimeError(
                f"Characterization interrupted; executor shutdown. "
                f"Unfinished task IDs: {unfinished}"
            )

        if capture:
            # Aggregate per-worker and per-batch metrics in a single linear pass.
            worker_pids = sorted({
                r.worker_pid for r in all_batch_results
                if r.worker_pid is not None
            })
            per_worker_task_count = {
                str(pid): sum(
                    len(r.task_results)
                    for r in all_batch_results
                    if r.worker_pid == pid
                )
                for pid in worker_pids
            }
            per_worker_batch_count = {
                str(pid): sum(
                    1 for r in all_batch_results if r.worker_pid == pid
                )
                for pid in worker_pids
            }
            per_worker_wall_seconds = {
                str(pid): sum(
                    r.wall_seconds for r in all_batch_results
                    if r.worker_pid == pid and r.wall_seconds is not None
                )
                for pid in worker_pids
            }
            batch_records = [
                {
                    'batch_id': r.batch_id,
                    'worker_pid': r.worker_pid,
                    'wall_seconds': r.wall_seconds,
                    'predicted_cost': r.predicted_cost,
                    'task_count': len(r.task_results),
                    'task_ids': r.task_ids,
                }
                for r in all_batch_results
            ]
            record_by_task_id = {record.task_id: record for record in records}
            task_records = [
                {
                    'task_id': tr.task_id,
                    'procedure': record_by_task_id[tr.task_id].procedure,
                    'cell': record_by_task_id[tr.task_id].cell,
                    'batch_id': r.batch_id,
                    'worker_pid': r.worker_pid,
                    'wall_seconds': tr.task_wall_seconds,
                    'error_type': tr.error_type,
                }
                for r in all_batch_results
                for tr in r.task_results
            ]
            metrics.worker_pids = worker_pids
            metrics.per_worker_task_count = per_worker_task_count
            metrics.per_worker_batch_count = per_worker_batch_count
            metrics.per_worker_wall_seconds = per_worker_wall_seconds
            metrics.batch_records = sorted(batch_records, key=lambda b: b['batch_id'])
            metrics.task_records = sorted(task_records, key=lambda t: t['task_id'])

        merge_start = time.perf_counter()
        self._apply_batched_results(
            all_results, records, self.settings.omit_on_failure
        )
        metrics.merge_seconds += time.perf_counter() - merge_start

    def characterize(self):
        """Execute scheduled simulation jobs in parallel"""
        # Wrap raw tasks in TaskRecord objects for scheduler tracking
        records = []
        task_index = 0
        for (cell, config) in self.cells:
            for (task_fn, *task_args) in self.analyse_cell(cell, config):
                proc = f"{task_fn.__module__}.{task_fn.__qualname__}"
                records.append(TaskRecord(
                    task_id=task_index,
                    callable=task_fn,
                    args=tuple(task_args),
                    procedure=proc,
                    cell=cell.name,
                ))
                task_index += 1

        # Scheduler metrics are collected but not used to alter output in S1
        metrics = SchedulerMetrics(
            task_count=len(records),
            future_count=len(records),
        )
        metrics.capture_metrics = self.settings.scheduler_metrics_path is not None

        # Run all simulation jobs and merge each resulting liberty cell group into the library
        with tqdm(bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                  total=len(records), desc="Characterizing") as progress_bar:
            if self.settings.simulation.execution_engine == 'batched':
                self._characterize_batched(records, progress_bar, metrics)
            else:
                metrics.requested_jobs = self.settings.jobs
                metrics.resolved_max_workers = self.settings.jobs or os.cpu_count() or 4
                metrics.worker_pids = []
                with ProcessPoolExecutor(max_workers=self.settings.jobs) as executor:
                    submit_start = time.perf_counter()
                    futures = [executor.submit(record.callable, *record.args) for record in records]
                    future_to_record = {future: record for future, record in zip(futures, records)}
                    metrics.submit_seconds = time.perf_counter() - submit_start

                    for future in as_completed(futures):
                        record = future_to_record[future]
                        try:
                            value = future.result()
                            result = TaskResult(task_id=record.task_id, value=value)
                        except Exception as e:
                            result = TaskResult(
                                task_id=record.task_id,
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                            if self.settings.omit_on_failure and isinstance(e, ProcedureFailedException):
                                continue
                            raise
                        merge_start = time.perf_counter()
                        self.library.add_group(result.value)
                        metrics.merge_seconds += time.perf_counter() - merge_start
                        progress_bar.update(1)

        # Post-processing: Fetch generated table templates and add them to the library
        lut_templates = []
        for timing_group in self.library.subgroups_with_name('timing'):
            lut_templates += [lut_group.template for lut_group in timing_group.groups.values()]
        [self.library.add_group(lut_template) for lut_template in lut_templates]

        liberty = self.library.to_liberty(precision=6)

        # Write scheduler metrics once, after merge is complete.
        if metrics.capture_metrics:
            metrics_path = Path(self.settings.scheduler_metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, sort_keys=True)

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
        return liberty


class CharacterizationSettings:
    """Container for characterization settings"""
    def __init__(self, **kwargs):
        """Create a new CharacterizationSettings instance"""
        # Behavioral settings
        self.jobs = None if kwargs.pop('multithreaded', True) else 1
        self.results_dir = Path(kwargs.pop('results_dir', 'results'))
        self.plots_dir = self.results_dir / 'plots'
        self.debug = kwargs.pop('debug', False)
        self.debug_dir = Path(kwargs.pop('debug_dir', 'debug'))
        self.quiet = kwargs.pop('quiet', False)
        self.cell_defaults = kwargs.get('cell_defaults', {})
        self.omit_on_failure = kwargs.get('omit_on_failure', False)
        self.scheduler_metrics_path = kwargs.get('scheduler_metrics_path', None)

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
            'output_threshold_pct_fall': self.logic_thresholds.falling,
        }


class SimulationSettings:
    """Container for simulation backend and procedures"""
    def __init__(self, **kwargs):
        self.backend = kwargs.get('backend', 'ngspice-shared')
        self.input_capacitance = registered_procedures[
            kwargs.get('input_capacitance_procedure', 'ac_sweep')
        ]["callable"]
        self.combinational_delay = registered_procedures[
            kwargs.get('combinational_delay_procedure', 'combinational_worst_case')
        ]["callable"]
        self.combinational_leakage = registered_procedures[
            kwargs.get('combinational_leakage_procedure', 'combinational_leakage')
        ]["callable"]
        self.sequential_delay = registered_procedures[
            kwargs.get('sequential_delay_procedure', 'sequential_worst_case')
        ]["callable"]
        self.metastability_constraint = registered_procedures[
            kwargs.get('setup_hold_constraint_procedure', 'measure_setup_hold_from_contour')
        ]["callable"]
        self.recovery_constraint = registered_procedures[
            kwargs.get('recovery_constraint_procedure', 'recovery_constraint')
        ]["callable"]
        self.removal_constraint = registered_procedures[
            kwargs.get('removal_constraint_procedure', 'removal_constraint')
        ]["callable"]
        self.min_pulse_width_constraint = registered_procedures[
            kwargs.get('min_pulse_width_constraint_procedure', 'min_pulse_width_constraint')
        ]["callable"]

        # Scheduler configuration (S3)
        self.execution_engine = kwargs.get('execution_engine', "legacy")
        self.scheduler_batch_factor = kwargs.get('scheduler_batch_factor', 4)
        self.scheduler_max_inflight_per_worker = kwargs.get('scheduler_max_inflight_per_worker', 2)
        self.scheduler_cost_policy = kwargs.get('scheduler_cost_policy', "measured_lpt")

        if self.execution_engine not in ("legacy", "batched"):
            raise ValueError(
                f"Invalid execution_engine: {self.execution_engine!r}; "
                f"must be 'legacy' or 'batched'"
            )
        if self.scheduler_batch_factor < 1:
            raise ValueError(
                f"scheduler_batch_factor must be >= 1, got {self.scheduler_batch_factor}"
            )
        if self.scheduler_max_inflight_per_worker < 1:
            raise ValueError(
                f"scheduler_max_inflight_per_worker must be >= 1, got {self.scheduler_max_inflight_per_worker}"
            )
        if self.scheduler_cost_policy != "measured_lpt":
            raise ValueError(
                f"Invalid scheduler_cost_policy: {self.scheduler_cost_policy!r}; "
                f"must be 'measured_lpt'"
            )


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
