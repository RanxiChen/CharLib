"""Pure-data IR for reusable ngspice execution."""

import time
import os as _os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class SimulationPoint:
    """Immutable point description. Primitives/tuples/str/numbers only."""
    point_id: str
    task_id: int
    cell_name: str
    netlist_path: str
    model_paths: Tuple[Tuple[str, str], ...]
    pin_order: Tuple[str, ...]
    input_pin: str
    output_pin: str
    input_transition: str
    output_transition: str
    stable_inputs: Tuple[str, ...]
    ignored_outputs: Tuple[str, ...]
    data_slew: float
    load: float
    t_sim_end: float
    temperature: float
    supplies: Tuple[Tuple[str, float], ...]
    thresholds: Tuple[float, float]
    time_unit_str: str
    voltage_unit_str: str
    criterion_group: str
    state_condition: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class TopologySignature:
    """Circuit topology fingerprint. Includes swept values that affect netlist
    topology (data_slew, t_sim_end). Excludes alterable sweep values (load,
    temperature)."""
    cell_hash: str
    netlist_hash: str
    model_hashes: Tuple[Tuple[str, str], ...]
    pin_topology: Tuple[Tuple[str, str, str], ...]
    state_condition: Tuple[Tuple[str, str], ...]
    measurement_names: Tuple[str, ...]
    measurement_directions: Tuple[str, ...]
    backend: str
    temperature: float
    supplies_hash: str
    data_slew: float = 0.0
    t_sim_end: float = 0.0

    def __hash__(self):
        return hash((
            self.cell_hash, self.netlist_hash, self.model_hashes,
            self.pin_topology, self.state_condition, self.measurement_names,
            self.measurement_directions, self.backend, self.data_slew,
            self.t_sim_end, self.supplies_hash
        ))


@dataclass
class MeasurementResult:
    """One point execution result."""
    point_id: str
    task_id: int
    signature: Optional[TopologySignature] = None
    measurement: Dict[str, float] = field(default_factory=dict)
    status: str = 'ok'
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    fresh_retry: bool = False
    deck_load_count: int = 1


class LibertyAssembler:
    """Deterministic collector. Receives MeasurementResults, orders by point_id."""

    def __init__(self):
        self._results: list = []

    def add_results(self, results: list):
        self._results.extend(results)

    def ordered_results(self) -> list:
        return sorted(self._results, key=lambda r: r.point_id)


class WorkerContext:
    """Single-process ngspice context for signature-based reuse.

    One context owns one NgSpiceShared instance and at most one loaded TopologySignature.
    On signature match, alters sweep values and re-runs. On mismatch or error, reloads.

    Signature includes: cell_hash, netlist_hash, model_hashes, pin_topology, state_condition,
                         measurement_names, measurement_directions, backend, data_slew, t_sim_end, supplies_hash.
    Signature excludes: load (alterable via alter cy c=...), temperature (alterable via set temp=...).
    """

    def __init__(self):
        self._shared = None
        self._current_signature = None
        self._quarantined_signatures = set()
        self._deck_load_count = 0

    @property
    def is_loaded(self):
        return self._shared is not None and self._current_signature is not None

    @property
    def deck_load_count(self):
        return self._deck_load_count

    def _signature_load_key(self, signature):
        """Compute a hashable key for signature matching.
        Excludes alterable fields: load, temperature."""
        return (getattr(signature, 'cell_hash', ''),
                getattr(signature, 'netlist_hash', ''),
                getattr(signature, 'model_hashes', ()),
                getattr(signature, 'pin_topology', ()),
                getattr(signature, 'state_condition', ()),
                getattr(signature, 'measurement_names', ()),
                getattr(signature, 'measurement_directions', ()),
                getattr(signature, 'backend', ''),
                getattr(signature, 'data_slew', 0.0),
                getattr(signature, 't_sim_end', 0.0),
                getattr(signature, 'supplies_hash', ''))

    def _signature_matches(self, signature):
        if not self.is_loaded:
            return False
        if self._signature_load_key(signature) in self._quarantined_signatures:
            return False
        return self._signature_load_key(signature) == self._signature_load_key(self._current_signature)

    def ensure_loaded(self, deck_text, signature):
        """Load deck if signature doesn't match. Returns True if new load.

        PySpice/ngspice-shared allows only one NgSpiceShared object per process, so
        signature switches reload the deck into the existing shared object instead
        of destroying it and constructing another instance.
        """
        if self._signature_matches(signature):
            return False
        try:
            if self._shared is None:
                from PySpice.Spice.NgSpice.Shared import NgSpiceShared
                self._shared = NgSpiceShared()
            self._shared.load_circuit(deck_text)
            self._current_signature = signature
            self._deck_load_count += 1
            return True
        except Exception:
            self.discard()
            raise

    def alter_sweep_values(self, load, temperature, output_pin='Y'):
        """Apply alterable sweep values. Must call after ensure_loaded."""
        if not self.is_loaded:
            raise RuntimeError("No circuit loaded")
        self._shared.reset()
        self._shared.exec_command(f'alter c{str(output_pin).lower()} c={load}')
        self._shared.exec_command(f'set temp={temperature}')

    def run_and_extract(self):
        """Run and return measurement dict {name: float}."""
        if not self.is_loaded:
            raise RuntimeError("No circuit loaded")
        raw = self._shared.run()
        return {k: float(v) for k, v in raw.items()}

    def discard(self):
        """Clear loaded signature. Keep the one shared object alive for reuse."""
        if self._shared is not None:
            try:
                self._shared.reset()
            except Exception:
                pass
        self._current_signature = None

    def quarantine_signature(self, signature):
        """Mark signature as quarantined."""
        key = self._signature_load_key(signature)
        self._quarantined_signatures.add(key)
        if self.is_loaded and self._signature_load_key(self._current_signature) == key:
            self.discard()

    def execute_point_with_context(self, point, deck_text, signature, measurement_names,
                                   fresh_retry=None):
        """Execute one point using the reusable ngspice context.

        Returns a MeasurementResult. On load/alter/run/extract error, quarantine
        the signature and retry exactly once through the supplied fresh path.
        """
        start = time.perf_counter()
        try:
            loaded = self.ensure_loaded(deck_text, signature)
            self.alter_sweep_values(point.load, point.temperature, point.output_pin)
            raw_measurements = self.run_and_extract()
            measurement_names = tuple(measurement_names)
            measurements = {name: float(raw_measurements[name])
                            for name in measurement_names
                            if name in raw_measurements}
            return MeasurementResult(
                point_id=point.point_id,
                task_id=point.task_id,
                signature=signature,
                measurement=measurements,
                status='ok',
                timings={'total': time.perf_counter() - start},
                fresh_retry=False,
                deck_load_count=1 if loaded else 0,
            )
        except Exception as exc:
            self.quarantine_signature(signature)
            if fresh_retry is not None:
                retry_result = fresh_retry()
                retry_result.fresh_retry = True
                return retry_result
            return MeasurementResult(
                point_id=point.point_id,
                task_id=point.task_id,
                signature=signature,
                status='error',
                error=str(exc),
                timings={'total': time.perf_counter() - start},
                fresh_retry=False,
                deck_load_count=1,
            )


@dataclass
class WorkerRequest:
    """Pure-data request sent from parent to worker process.
    Contains ONLY primitives, tuples, and SimulationPoint references.
    NO Cell, Config, Settings, PySpice, Liberty, or NgSpiceShared objects."""
    request_id: str
    point: SimulationPoint
    deck_text: str
    signature: TopologySignature


@dataclass
class WorkerResult:
    """Pure-data result sent from worker back to parent.
    Contains ONLY MeasurementResult and status.
    NO PySpice, Liberty, or NgSpiceShared objects."""
    request_id: str
    result: MeasurementResult
    worker_id: int = 0
    worker_deck_load_count: int = 0
    worker_error: Optional[str] = None


_worker_context: Optional[WorkerContext] = None
_worker_id: int = -1


def init_worker(worker_id: int = -1):
    """Module-level initializer called once per ProcessPoolExecutor worker.

    Creates the worker's own WorkerContext and NgSpiceShared instance.
    Must be called before any execute_request calls.
    """
    global _worker_context, _worker_id
    _worker_context = WorkerContext()
    _worker_id = worker_id


def get_worker_context() -> Optional[WorkerContext]:
    """Return the current worker's context, or None if not initialized."""
    return _worker_context


def get_worker_id() -> int:
    """Return the current worker's id, or -1 if not initialized."""
    return _worker_id


def shutdown_worker():
    """Clean up the worker's context."""
    global _worker_context, _worker_id
    if _worker_context is not None:
        _worker_context.discard()
    _worker_context = None
    _worker_id = -1


def detect_missing_requests(requests, results):
    """Return request_ids present in requests but absent from results.

    The returned list preserves the order of the input requests.
    """
    result_ids = {result.request_id for result in results}
    return [request.request_id for request in requests if request.request_id not in result_ids]


def execute_request(request: WorkerRequest) -> WorkerResult:
    """Execute a WorkerRequest using the per-process WorkerContext.

    This is the ONLY function called by the parent process via ProcessPoolExecutor.
    It uses the module-level WorkerContext created by init_worker().
    """
    global _worker_context
    if request is None or request.point is None:
        return WorkerResult(
            request_id=getattr(request, 'request_id', 'unknown'),
            result=MeasurementResult(
                point_id=getattr(request.point, 'point_id', None) if request else None,
                task_id=getattr(request.point, 'task_id', -1) if request else -1,
                status='error',
                error='Invalid request: point is None',
            ),
            worker_id=_worker_id,
            worker_error='Invalid request: point is None',
        )
    if _worker_context is None:
        return WorkerResult(
            request_id=request.request_id,
            result=MeasurementResult(
                point_id=request.point.point_id,
                task_id=request.point.task_id,
                status='error',
                error='Worker not initialized',
            ),
            worker_id=_worker_id,
            worker_error='Worker not initialized',
        )
    try:
        was_loaded = _worker_context.ensure_loaded(request.deck_text, request.signature)
        _worker_context.alter_sweep_values(request.point.load, request.point.temperature)
        measurements = _worker_context.run_and_extract()

        mresult = MeasurementResult(
            point_id=request.point.point_id,
            task_id=request.point.task_id,
            signature=request.signature,
            measurement=measurements,
            status='ok',
            deck_load_count=1 if was_loaded else 0,
        )

        return WorkerResult(
            request_id=request.request_id,
            result=mresult,
            worker_id=_worker_id,
            worker_deck_load_count=_worker_context.deck_load_count,
        )
    except Exception as e:
        return WorkerResult(
            request_id=request.request_id,
            result=MeasurementResult(
                point_id=request.point.point_id,
                task_id=request.point.task_id,
                status='error',
                error=str(e),
            ),
            worker_id=_worker_id,
            worker_deck_load_count=_worker_context.deck_load_count if _worker_context else 0,
            worker_error=str(e),
        )
