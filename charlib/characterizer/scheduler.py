"""Scheduler data structures and instrumentation for CharLib characterization tasks."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TaskRecord:
    """A wrapper around an original simulation task so the scheduler can track it."""
    task_id: int              # zero-based position in simulation_tasks
    callable: Callable        # original callable object
    args: tuple               # original args tuple
    procedure: str            # from callable.__module__ / __qualname__
    cell: str = "unknown"     # cell name if known
    cost_hint: float = 1.0    # predicted relative cost

    def __post_init__(self):
        # procedure is derived from the callable; ensure it is set even if passed empty
        if not self.procedure and self.callable is not None:
            module = getattr(self.callable, '__module__', '<unknown>')
            qualname = getattr(self.callable, '__qualname__', '<unknown>')
            object.__setattr__(self, 'procedure', f'{module}.{qualname}')


@dataclass
class TaskResult:
    """Result of executing a single TaskRecord."""
    task_id: int
    value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional['SchedulerMetrics'] = None


@dataclass
class SchedulerMetrics:
    """Aggregate scheduler timing/throughput metrics."""
    task_count: int = 0
    future_count: int = 0
    batch_count: int = 0
    submit_seconds: float = 0.0
    collect_seconds: float = 0.0
    merge_seconds: float = 0.0
    ipc_estimate_bytes: int = 0
