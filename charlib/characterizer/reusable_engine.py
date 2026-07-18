"""Pure-data IR for reusable ngspice execution."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import hashlib


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
        """Load deck if signature doesn't match. Returns True if new load."""
        if self._signature_matches(signature):
            return False
        if self._shared is not None:
            try:
                self._shared.destroy()
            except Exception:
                pass
        try:
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            self._shared = NgSpiceShared()
            self._shared.load_circuit(deck_text)
            self._current_signature = signature
            self._deck_load_count += 1
            return True
        except Exception:
            self.discard()
            raise

    def alter_sweep_values(self, load, temperature):
        """Apply alterable sweep values. Must call after ensure_loaded."""
        if not self.is_loaded:
            raise RuntimeError("No circuit loaded")
        self._shared.reset()
        self._shared.exec_command(f'alter cy c={load}')
        self._shared.exec_command(f'set temp={temperature}')

    def run_and_extract(self):
        """Run and return measurement dict {name: float}."""
        if not self.is_loaded:
            raise RuntimeError("No circuit loaded")
        raw = self._shared.run()
        return {k: float(v) for k, v in raw.items()}

    def discard(self):
        """Destroy context. Subsequent uses force fresh loads."""
        if self._shared is not None:
            try:
                self._shared.destroy()
            except Exception:
                pass
        self._shared = None
        self._current_signature = None

    def quarantine_signature(self, signature):
        """Mark signature as quarantined."""
        key = self._signature_load_key(signature)
        self._quarantined_signatures.add(key)
        if self.is_loaded and self._signature_load_key(self._current_signature) == key:
            self.discard()
