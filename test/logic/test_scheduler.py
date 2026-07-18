import os
import pickle
import operator
from concurrent.futures import ProcessPoolExecutor
import pytest
from charlib.characterizer.scheduler import (
    TaskRecord,
    TaskResult,
    SchedulerMetrics,
    BatchRecord,
    BatchResult,
    plan_batches,
    execute_batch,
)


def dummy_double(x):
    return x * 2


def dummy_fail(x):
    raise ValueError(f"bad input: {x}")


class TestTaskRecord:
    def test_stable_id(self):
        r = TaskRecord(task_id=5, callable=dummy_double, args=(3,), procedure="test.dummy_double")
        assert r.task_id == 5

    def test_procedure_auto_derive(self):
        r = TaskRecord(task_id=0, callable=dummy_double, args=(), procedure="")
        assert "dummy_double" in r.procedure

    def test_picklable(self):
        r = TaskRecord(task_id=1, callable=dummy_double, args=(2,), procedure="test.dummy_double")
        data = pickle.dumps(r)
        r2 = pickle.loads(data)
        assert r2.task_id == 1
        assert r2.args == (2,)


class TestTaskResult:
    def test_success(self):
        r = TaskResult(task_id=0, value=42)
        assert r.task_id == 0
        assert r.value == 42
        assert r.error_type is None

    def test_error(self):
        r = TaskResult(task_id=1, error_type="ValueError", error_message="bad input")
        assert r.task_id == 1
        assert r.error_type == "ValueError"
        assert r.value is None

    def test_picklable(self):
        r = TaskResult(task_id=2, value="hello")
        data = pickle.dumps(r)
        r2 = pickle.loads(data)
        assert r2.value == "hello"


class TestSchedulerMetrics:
    def test_defaults(self):
        m = SchedulerMetrics()
        assert m.task_count == 0
        assert m.future_count == 0

    def test_serialization(self):
        m = SchedulerMetrics(task_count=10, future_count=10, submit_seconds=1.5)
        d = {"task_count": m.task_count, "future_count": m.future_count, "batch_count": m.batch_count,
             "submit_seconds": m.submit_seconds, "collect_seconds": m.collect_seconds,
             "merge_seconds": m.merge_seconds, "ipc_estimate_bytes": m.ipc_estimate_bytes}
        assert d["task_count"] == 10
        assert d["submit_seconds"] == 1.5


class TestPlanBatches:
    def _records(self, costs):
        return [
            TaskRecord(task_id=i, callable=dummy_double, args=(i,), procedure="test.dummy_double", cost_hint=c)
            for i, c in enumerate(costs)
        ]

    def test_empty(self):
        assert plan_batches([], workers=4) == []

    def test_single_task_one_batch(self):
        records = [
            TaskRecord(task_id=0, callable=dummy_double, args=(5,), procedure="test.dummy_double", cost_hint=10.0)
        ]
        batches = plan_batches(records, workers=4)
        assert len(batches) == 1
        assert batches[0].tasks[0].task_id == 0

    def test_max_one_task_per_batch_for_many_workers(self):
        records = self._records([1.0, 2.0, 3.0])
        batches = plan_batches(records, workers=100)
        assert len(batches) == len(records)

    def test_no_empty_batches(self):
        records = self._records([1.0, 2.0, 3.0, 4.0, 5.0])
        batches = plan_batches(records, workers=2)
        assert all(len(b.tasks) > 0 for b in batches)

    def test_all_task_ids_present(self):
        records = self._records([3.0, 1.0, 4.0, 2.0])
        batches = plan_batches(records, workers=2)
        ids = sorted(t.task_id for b in batches for t in b.tasks)
        assert ids == list(range(len(records)))

    def test_batch_ids_sequential(self):
        records = self._records([1.0, 2.0, 3.0, 4.0, 5.0])
        batches = plan_batches(records, workers=2)
        assert [b.batch_id for b in batches] == list(range(len(batches)))

    def test_heavy_tasks_split(self):
        records = [
            TaskRecord(task_id=0, callable=dummy_double, args=(0,), procedure="t", cost_hint=100.0),
            TaskRecord(task_id=1, callable=dummy_double, args=(1,), procedure="t", cost_hint=90.0),
            TaskRecord(task_id=2, callable=dummy_double, args=(2,), procedure="t", cost_hint=1.0),
        ]
        batches = plan_batches(records, workers=2)
        ids_per_batch = [tuple(sorted(t.task_id for t in b.tasks)) for b in batches]
        assert (0, 1) not in ids_per_batch

    def test_predicted_cost_is_sum(self):
        records = self._records([2.0, 3.0, 5.0])
        batches = plan_batches(records, workers=2)
        total = sum(b.predicted_cost for b in batches)
        assert total == pytest.approx(10.0)


class TestExecuteBatch:
    def test_success(self):
        records = [
            TaskRecord(task_id=0, callable=dummy_double, args=(3,), procedure="t", cost_hint=1.0),
            TaskRecord(task_id=1, callable=dummy_double, args=(4,), procedure="t", cost_hint=1.0),
        ]
        batch = BatchRecord(batch_id=7, tasks=records, predicted_cost=2.0)
        result = execute_batch(batch)
        assert result.batch_id == 7
        assert len(result.task_results) == 2
        assert result.task_results[0].value == 6
        assert result.task_results[1].value == 8
        assert result.worker_pid == os.getpid()
        assert result.wall_seconds >= 0.0

    def test_failure_captured(self):
        record = TaskRecord(task_id=5, callable=dummy_fail, args=(7,), procedure="t", cost_hint=1.0)
        batch = BatchRecord(batch_id=0, tasks=[record], predicted_cost=1.0)
        result = execute_batch(batch)
        assert result.batch_id == 0
        assert len(result.task_results) == 1
        assert result.task_results[0].error_type == "ValueError"
        assert "bad input: 7" in result.task_results[0].error_message


class TestBatchPickling:
    def test_batch_record_pickles(self):
        records = [
            TaskRecord(task_id=0, callable=dummy_double, args=(2,), procedure="t", cost_hint=1.0)
        ]
        batch = BatchRecord(batch_id=3, tasks=records, predicted_cost=1.0)
        data = pickle.dumps(batch)
        b2 = pickle.loads(data)
        assert b2.batch_id == 3
        assert b2.predicted_cost == 1.0
        assert b2.tasks[0].args == (2,)

    def test_batch_result_pickles(self):
        results = [TaskResult(task_id=1, value="ok")]
        br = BatchResult(batch_id=2, task_results=results, worker_pid=123, wall_seconds=0.5)
        data = pickle.dumps(br)
        br2 = pickle.loads(data)
        assert br2.batch_id == 2
        assert br2.task_results[0].value == "ok"
        assert br2.worker_pid == 123
        assert br2.wall_seconds == 0.5


class TestProcessPoolBatching:
    def test_pool_executes_batches(self):
        batch1 = BatchRecord(
            batch_id=0,
            tasks=[
                TaskRecord(task_id=0, callable=operator.add, args=(1, 1), procedure="operator.add", cost_hint=1.0),
                TaskRecord(task_id=1, callable=operator.add, args=(2, 2), procedure="operator.add", cost_hint=1.0),
            ],
            predicted_cost=2.0,
        )
        batch2 = BatchRecord(
            batch_id=1,
            tasks=[
                TaskRecord(task_id=2, callable=operator.add, args=(3, 3), procedure="operator.add", cost_hint=1.0),
            ],
            predicted_cost=1.0,
        )
        with ProcessPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(execute_batch, batch1)
            future2 = executor.submit(execute_batch, batch2)
            results = [future1.result(), future2.result()]
        values = sorted(
            (r.task_id, r.value)
            for batch_result in results
            for r in batch_result.task_results
        )
        assert values == [(0, 2), (1, 4), (2, 6)]
