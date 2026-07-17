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
    """Circuit topology fingerprint. Excludes swept values (slew, load, sim_end)."""
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

    def __hash__(self):
        return hash((
            self.cell_hash, self.netlist_hash, self.model_hashes,
            self.pin_topology, self.state_condition, self.measurement_names,
            self.measurement_directions, self.backend, self.temperature,
            self.supplies_hash
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
