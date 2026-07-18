import os
import operator
import time
import json
import signal
import tempfile
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pytest
import yaml

from charlib.config.syntax import ConfigFile
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


def dummy_slow(x, duration=0.01):
    time.sleep(duration)
    return Group('cell', f'cell_{x}')


class TestTaskRecord:
    def test_stable_id(self):
        r = TaskRecord(task_id=5, callable=dummy_double, args=(3,), procedure="test.dummy_double")
        assert r.task_id == 5

    def test_procedure_auto_derive(self):
        r = TaskRecord(task_id=0, callable=dummy_double, args=(), procedure="")
        assert "dummy_double" in r.procedure

    def test_picklable(self):
        r = TaskRecord(task_id=1, callable=dummy_double, args=(2,), procedure="test.dummy_double")
        import pickle
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
        import pickle
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
        import pickle
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
        import pickle
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


# ---- S3 tests -----------------------------------------------------------------

from charlib.characterizer.characterizer import Characterizer, SimulationSettings
from charlib.liberty.liberty import Group


class TestExecutionEngineConfig:
    def test_default_is_legacy(self):
        s = SimulationSettings()
        assert s.execution_engine == "legacy"

    def test_valid_batched(self):
        s = SimulationSettings(execution_engine="batched")
        assert s.execution_engine == "batched"

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid execution_engine"):
            SimulationSettings(execution_engine="neither")

    def test_batch_factor_validation(self):
        with pytest.raises(ValueError, match="scheduler_batch_factor"):
            SimulationSettings(scheduler_batch_factor=0)

    def test_inflight_validation(self):
        with pytest.raises(ValueError, match="scheduler_max_inflight_per_worker"):
            SimulationSettings(scheduler_max_inflight_per_worker=0)

    def test_cost_policy_validation(self):
        with pytest.raises(ValueError, match="scheduler_cost_policy"):
            SimulationSettings(scheduler_cost_policy="random")


class TestYAMLExecutionEngineConfig:
    """YAML round-trip tests for simulation execution_engine and batch config."""

    W1_YAML_PATH = Path(__file__).parent.parent / 'pdks' / 'ex_osu350_invx1.yaml'

    @pytest.fixture
    def w1_config(self):
        with open(self.W1_YAML_PATH) as f:
            return yaml.safe_load(f)

    def test_yaml_batched_engine_passes(self, w1_config):
        """A real YAML with execution_engine=batched validates and is wired to SimulationSettings."""
        w1_config['settings']['simulation'] = {
            'execution_engine': 'batched',
            'scheduler_batch_factor': 4,
            'scheduler_max_inflight_per_worker': 2,
        }
        validated = ConfigFile.validate(w1_config)
        sim = validated['settings']['simulation']
        assert sim['execution_engine'] == 'batched'
        settings = SimulationSettings(**sim)
        assert settings.execution_engine == 'batched'
        assert settings.scheduler_batch_factor == 4
        assert settings.scheduler_max_inflight_per_worker == 2

    def test_yaml_missing_simulation_defaults_to_legacy(self, w1_config):
        """A real YAML without a simulation section defaults to legacy execution."""
        w1_config['settings'].pop('simulation', None)
        validated = ConfigFile.validate(w1_config)
        assert 'simulation' not in validated['settings']
        settings = SimulationSettings(**validated['settings'].get('simulation', {}))
        assert settings.execution_engine == 'legacy'

    def test_yaml_invalid_engine_raises(self, w1_config):
        """An invalid execution_engine in YAML raises ValueError."""
        w1_config['settings']['simulation'] = {'execution_engine': 'invalid'}
        with pytest.raises(ValueError, match="Invalid execution_engine"):
            ConfigFile.validate(w1_config)

    def test_yaml_batch_factor_preserved(self, w1_config):
        """scheduler_batch_factor is preserved through ConfigFile.validate."""
        w1_config['settings']['simulation'] = {'scheduler_batch_factor': 8}
        validated = ConfigFile.validate(w1_config)
        sim = validated['settings']['simulation']
        assert sim['scheduler_batch_factor'] == 8

    def test_yaml_round_trip(self, w1_config):
        """YAML -> ConfigFile -> YAML preserves all simulation fields."""
        w1_config['settings']['simulation'] = {
            'execution_engine': 'batched',
            'scheduler_batch_factor': 8,
            'scheduler_max_inflight_per_worker': 4,
            'scheduler_cost_policy': 'measured_lpt',
        }
        validated = ConfigFile.validate(w1_config)
        dumped = yaml.safe_dump(validated, sort_keys=False)
        reparsed = yaml.safe_load(dumped)
        sim = reparsed['settings']['simulation']
        assert sim['execution_engine'] == 'batched'
        assert sim['scheduler_batch_factor'] == 8
        assert sim['scheduler_max_inflight_per_worker'] == 4
        assert sim['scheduler_cost_policy'] == 'measured_lpt'


class TestDeterministicMerge:
    def _make_results(self):
        return [
            TaskResult(task_id=2, value=Group('cell', 'c')),
            TaskResult(task_id=0, value=Group('cell', 'a')),
            TaskResult(task_id=1, value=Group('cell', 'b')),
        ]

    def test_sort_by_task_id(self):
        c = Characterizer(lib_name='test')
        c._apply_batched_results(self._make_results(), records=[None, None, None], omit_on_failure=False)
        cells = list(c.library.subgroups_with_name('cell'))
        assert [g.identifier for g in cells] == ['a', 'b', 'c']

    def test_reject_duplicate_ids(self):
        c = Characterizer(lib_name='test')
        results = [
            TaskResult(task_id=0, value=Group('cell', 'a')),
            TaskResult(task_id=0, value=Group('cell', 'b')),
        ]
        with pytest.raises(RuntimeError, match="Duplicate task IDs"):
            c._apply_batched_results(results, records=[None, None], omit_on_failure=False)

    def test_reject_missing_ids(self):
        c = Characterizer(lib_name='test')
        results = [
            TaskResult(task_id=0, value=Group('cell', 'a')),
            TaskResult(task_id=2, value=Group('cell', 'c')),
        ]
        with pytest.raises(RuntimeError, match="Task ID mismatch"):
            c._apply_batched_results(results, records=[None, None, None], omit_on_failure=False)


class TestOmitOnFailure:
    def _make_results(self):
        return [
            TaskResult(task_id=0, value=Group('cell', 'a')),
            TaskResult(task_id=1, error_type='ValueError', error_message='bad', exception=ValueError('bad')),
            TaskResult(task_id=2, value=Group('cell', 'c')),
        ]

    def test_omit_true_skips_failed(self):
        c = Characterizer(lib_name='test')
        c._apply_batched_results(self._make_results(), records=[None, None, None], omit_on_failure=True)
        cells = list(c.library.subgroups_with_name('cell'))
        assert len(cells) == 2
        assert {g.identifier for g in cells} == {'a', 'c'}

    def test_omit_false_raises(self):
        c = Characterizer(lib_name='test')
        with pytest.raises(ValueError, match='bad'):
            c._apply_batched_results(self._make_results(), records=[None, None, None], omit_on_failure=False)

    def test_siblings_survive(self):
        c = Characterizer(lib_name='test')
        c._apply_batched_results(self._make_results(), records=[None, None, None], omit_on_failure=True)
        cells = list(c.library.subgroups_with_name('cell'))
        assert len(cells) == 2


class TestInterruptCleanup:
    def test_unfinished_task_ids(self):
        c = Characterizer(lib_name='test')
        records = [TaskRecord(task_id=i, callable=dummy_double, args=(i,), procedure="t") for i in range(5)]
        results = [TaskResult(task_id=0, value='a'), TaskResult(task_id=2, value='c')]
        assert c._unfinished_task_ids(records, results) == [1, 3, 4]

    def test_no_child_processes(self):
        from charlib.characterizer.characterizer import SchedulerMetrics as _SchedulerMetrics
        c = Characterizer(lib_name='test', simulation={'execution_engine': 'batched'}, jobs=2)
        records = [
            TaskRecord(task_id=i, callable=dummy_slow, args=(i, 0.01), procedure="t", cost_hint=1.0)
            for i in range(4)
        ]
        # dummy_slow returns a Group so _apply_batched_results can add it
        metrics = _SchedulerMetrics(task_count=4)

        class _Bar:
            def update(self, n):
                pass

        c._characterize_batched(records, _Bar(), metrics)
        # Give the OS a moment to reap worker processes
        time.sleep(0.2)
        assert len(multiprocessing.active_children()) == 0


class TestLegacyFallback:
    def test_legacy_path_unchanged(self):
        c = Characterizer(lib_name='test', simulation={'execution_engine': 'legacy'})
        assert c.settings.simulation.execution_engine == 'legacy'

    def test_explicit_batched_selection(self):
        c = Characterizer(lib_name='test', simulation={'execution_engine': 'batched'})
        assert c.settings.simulation.execution_engine == 'batched'


class TestMetricsSeparation:
    def test_metrics_not_in_liberty(self):
        with tempfile.TemporaryDirectory() as td:
            metrics_path = os.path.join(td, 'metrics.json')
            c = Characterizer(
                lib_name='test',
                simulation={'execution_engine': 'batched'},
                scheduler_metrics_path=metrics_path,
            )
            liberty = c.characterize()
            assert os.path.exists(metrics_path)
            with open(metrics_path) as f:
                data = json.load(f)
            assert data['task_count'] == 0
            assert data['batch_count'] == 0
            assert 'task_count' not in liberty
            assert 'batch_count' not in liberty


class TestExceptionPropagation:
    def test_execute_batch_captures_exception_object(self):
        record = TaskRecord(task_id=5, callable=dummy_fail, args=(7,), procedure="t", cost_hint=1.0)
        batch = BatchRecord(batch_id=0, tasks=[record], predicted_cost=1.0)
        result = execute_batch(batch)
        assert result.task_results[0].exception is not None
        assert isinstance(result.task_results[0].exception, ValueError)
        assert "bad input: 7" in str(result.task_results[0].exception)

    def test_apply_batched_results_re_raises_original(self):
        c = Characterizer(lib_name='test')
        exc = ValueError('original')
        results = [
            TaskResult(task_id=0, value=Group('cell', 'a')),
            TaskResult(task_id=1, error_type='ValueError', error_message='original', exception=exc),
            TaskResult(task_id=2, value=Group('cell', 'c')),
        ]
        with pytest.raises(ValueError, match='original') as exc_info:
            c._apply_batched_results(results, records=[None, None, None], omit_on_failure=False)
        assert exc_info.value is exc


class TestSignalInterruption:
    def test_batched_runs_in_non_main_thread(self):
        # Verify the SIGTERM handler guard allows _characterize_batched to run in a
        # non-main thread without failing on signal.signal.
        import threading
        c = Characterizer(lib_name='test', simulation={'execution_engine': 'batched'}, jobs=2)
        records = [
            TaskRecord(task_id=i, callable=dummy_slow, args=(i, 0.01), procedure="t", cost_hint=1.0)
            for i in range(4)
        ]
        from charlib.characterizer.characterizer import SchedulerMetrics
        metrics = SchedulerMetrics(task_count=4)

        class _Bar:
            def update(self, n):
                pass

        result = []

        def target():
            c._characterize_batched(records, _Bar(), metrics)
            result.append(True)

        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive()
        assert result


class TestWorkerScaling:
    def _run_batched(self, jobs):
        c = Characterizer(lib_name='test', simulation={'execution_engine': 'batched'}, jobs=jobs)
        records = [
            TaskRecord(task_id=i, callable=dummy_slow, args=(i, 0.01), procedure="t", cost_hint=1.0)
            for i in range(4)
        ]
        from charlib.characterizer.characterizer import SchedulerMetrics
        metrics = SchedulerMetrics(task_count=4)

        class _Bar:
            def update(self, n):
                pass

        c._characterize_batched(records, _Bar(), metrics)
        cells = list(c.library.subgroups_with_name('cell'))
        return sorted(g.identifier for g in cells)

    def test_j1_equals_j8(self):
        identifiers_j1 = self._run_batched(1)
        identifiers_j8 = self._run_batched(8)
        assert identifiers_j1 == ['cell_0', 'cell_1', 'cell_2', 'cell_3']
        assert identifiers_j8 == identifiers_j1
