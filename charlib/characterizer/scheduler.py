"""Scheduler data structures and instrumentation for CharLib characterization tasks."""

import os
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
    exception: Optional[Exception] = None
    task_wall_seconds: Optional[float] = None
    worker_pid: Optional[int] = None
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
    capture_metrics: bool = False
    requested_jobs: Optional[int] = None
    resolved_max_workers: int = 0
    max_in_flight: int = 0
    worker_pids: list = field(default_factory=list)
    per_worker_task_count: dict = field(default_factory=dict)
    per_worker_batch_count: dict = field(default_factory=dict)
    per_worker_wall_seconds: dict = field(default_factory=dict)
    batch_records: list = field(default_factory=list)
    task_records: list = field(default_factory=list)


@dataclass
class BatchRecord:
    """A group of tasks to be executed together by one worker."""
    batch_id: int
    tasks: list
    predicted_cost: float


@dataclass
class BatchResult:
    """Result of executing a batch."""
    batch_id: int
    task_results: list
    worker_pid: Optional[int] = None
    wall_seconds: Optional[float] = None
    predicted_cost: float = 0.0
    task_ids: list = field(default_factory=list)


def plan_batches(tasks: list, workers: int, batch_factor: int = 4) -> list:
    """Partition tasks into batches using cost-ordered LPT assignment."""
    n = len(tasks)
    if n == 0:
        return []
    min_batches = min(2 * workers, n)
    max_batches = min(4 * workers, n)
    target = max(min_batches, min(max_batches, batch_factor * workers))
    target = min(target, n)
    sorted_tasks = sorted(tasks, key=lambda t: (-t.cost_hint, t.task_id))
    batches = [BatchRecord(batch_id=i, tasks=[], predicted_cost=0.0) for i in range(target)]
    for task in sorted_tasks:
        lightest = min(batches, key=lambda b: (b.predicted_cost, b.batch_id))
        lightest.tasks.append(task)
        lightest.predicted_cost += task.cost_hint
    batches = [b for b in batches if b.tasks]
    for i, b in enumerate(batches):
        b.batch_id = i
    return batches


def execute_batch(batch: BatchRecord, capture: bool = False) -> BatchResult:
    """Execute all tasks in a batch sequentially. No nested pools/threads.

    When ``capture`` is False, no timing or PID information is collected,
    eliminating instrumentation overhead entirely.
    """
    if capture:
        return _execute_batch_instrumented(batch)
    return _execute_batch_fast(batch)


def _execute_batch_fast(batch: BatchRecord) -> BatchResult:
    """Execute a batch without any instrumentation overhead."""
    results = []
    for record in batch.tasks:
        try:
            value = record.callable(*record.args)
            results.append(TaskResult(task_id=record.task_id, value=value))
        except Exception as e:
            results.append(TaskResult(
                task_id=record.task_id,
                error_type=type(e).__name__,
                error_message=str(e),
                exception=e,
            ))
    return BatchResult(
        batch_id=batch.batch_id,
        task_results=results,
        predicted_cost=batch.predicted_cost,
        task_ids=[r.task_id for r in results],
    )


def _execute_batch_instrumented(batch: BatchRecord) -> BatchResult:
    """Execute a batch while recording per-task and per-batch wall-clock times."""
    import time as _time
    pid = os.getpid()
    t0 = _time.perf_counter()
    results = []
    for record in batch.tasks:
        t_task = _time.perf_counter()
        try:
            value = record.callable(*record.args)
            wall = _time.perf_counter() - t_task
            results.append(TaskResult(
                task_id=record.task_id,
                value=value,
                task_wall_seconds=wall,
                worker_pid=pid,
            ))
        except Exception as e:
            wall = _time.perf_counter() - t_task
            results.append(TaskResult(
                task_id=record.task_id,
                error_type=type(e).__name__,
                error_message=str(e),
                exception=e,
                task_wall_seconds=wall,
                worker_pid=pid,
            ))
    wall = _time.perf_counter() - t0
    return BatchResult(
        batch_id=batch.batch_id,
        task_results=results,
        worker_pid=pid,
        wall_seconds=wall,
        predicted_cost=batch.predicted_cost,
        task_ids=[r.task_id for r in results],
    )
