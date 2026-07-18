import pickle
import pytest
from charlib.characterizer.scheduler import TaskRecord, TaskResult, SchedulerMetrics


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
