"""Unit tests for the reusable-engine IR types.

These tests do not import charlib simulation logic, PySpice, ngspice, or a PDK.
"""

import dataclasses
import pickle
import sys
from typing import Tuple, get_origin, get_args

import pytest

from charlib.characterizer.reusable_engine import (
    SimulationPoint,
    TopologySignature,
    MeasurementResult,
    LibertyAssembler,
)


def _make_point(point_id="p1", load=2.0, temperature=25.0) -> SimulationPoint:
    return SimulationPoint(
        point_id=point_id,
        task_id=1,
        cell_name="INV",
        netlist_path="/tmp/inv.sp",
        model_paths=(("model", "/tmp/model.sp"),),
        pin_order=("A", "Y"),
        input_pin="A",
        output_pin="Y",
        input_transition="rise",
        output_transition="fall",
        stable_inputs=("A",),
        ignored_outputs=(),
        data_slew=1.0,
        load=load,
        t_sim_end=10.0,
        temperature=temperature,
        supplies=(("VDD", 1.8),),
        thresholds=(0.5, 0.5),
        time_unit_str="ns",
        voltage_unit_str="V",
        criterion_group="inv_rise",
        state_condition=(("A", "0"),),
    )


def test_simulation_point_pickle_roundtrip():
    point = _make_point()
    serialized = pickle.dumps(point)
    restored = pickle.loads(serialized)
    assert restored == point
    assert restored.point_id == "p1"


def test_topology_signature_hash_consistency():
    common = dict(
        cell_hash="abc",
        netlist_hash="def",
        model_hashes=(("m1", "h1"),),
        pin_topology=(("A", "Y", "inout"),),
        state_condition=(("A", "0"),),
        measurement_names=("delay",),
        measurement_directions=("rise",),
        backend="ngspice",
        temperature=25.0,
        supplies_hash="sup",
    )
    sig1 = TopologySignature(**common)
    sig2 = TopologySignature(**common)
    sig3 = TopologySignature(**{**common, "cell_hash": "xyz"})
    assert hash(sig1) == hash(sig2)
    assert hash(sig1) != hash(sig3)


def test_measurement_result_defaults():
    result = MeasurementResult(point_id="p1", task_id=1)
    assert result.deck_load_count == 1
    assert result.fresh_retry is False
    assert result.measurement == {}
    assert result.status == "ok"
    assert result.signature is None


def test_liberty_assembler_ordering():
    assembler = LibertyAssembler()
    r1 = MeasurementResult(point_id="point_b", task_id=2)
    r2 = MeasurementResult(point_id="point_a", task_id=1)
    r3 = MeasurementResult(point_id="point_c", task_id=3)
    assembler.add_results([r2, r3, r1])
    ordered = assembler.ordered_results()
    assert [r.point_id for r in ordered] == ["point_a", "point_b", "point_c"]


def _is_immutable_type(annotation):
    """Return True if the annotation denotes an immutable type."""
    if annotation in (str, int, float, bool, type(None)):
        return True
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        # Variadic tuple: Tuple[T, ...]
        if len(args) == 2 and args[1] is ...:
            return _is_immutable_type(args[0])
        return all(_is_immutable_type(arg) for arg in args)
    return False


def test_simulation_point_is_frozen_and_uses_immutable_types():
    assert SimulationPoint.__dataclass_params__.frozen is True
    for field in dataclasses.fields(SimulationPoint):
        assert _is_immutable_type(field.type), (
            f"SimulationPoint.{field.name} is annotated with mutable type {field.type}"
        )


def test_simulation_point_rejects_runtime_mutation():
    point = _make_point()
    with pytest.raises(dataclasses.FrozenInstanceError):
        point.point_id = "mutated"


def test_worker_context_deck_load_count():
    """Different data_slew produces different load keys (must enter signature)."""
    from charlib.characterizer.reusable_engine import WorkerContext, TopologySignature
    ctx = WorkerContext()
    assert ctx.deck_load_count == 0
    sig1 = TopologySignature(cell_hash='a', netlist_hash='b', model_hashes=(),
                             pin_topology=(), state_condition=(),
                             measurement_names=('cell_rise__a_to_y',),
                             measurement_directions=(), backend='ngspice',
                             data_slew=0.015, t_sim_end=3.0,
                             temperature=25.0, supplies_hash='c')
    sig2 = TopologySignature(cell_hash='a', netlist_hash='b', model_hashes=(),
                             pin_topology=(), state_condition=(),
                             measurement_names=('cell_rise__a_to_y',),
                             measurement_directions=(), backend='ngspice',
                             data_slew=0.030, t_sim_end=3.0,
                             temperature=25.0, supplies_hash='c')
    assert ctx._signature_load_key(sig1) != ctx._signature_load_key(sig2)


def test_worker_context_signature_switching():
    """Temperature difference excluded from signature → same load key."""
    from charlib.characterizer.reusable_engine import WorkerContext, TopologySignature
    ctx = WorkerContext()
    sig_a = TopologySignature(cell_hash='x', netlist_hash='y', model_hashes=(),
                              pin_topology=(), state_condition=(),
                              measurement_names=(), measurement_directions=(),
                              backend='ngspice', data_slew=0.015,
                              t_sim_end=3.0, temperature=25.0, supplies_hash='z')
    sig_b = TopologySignature(cell_hash='x', netlist_hash='y', model_hashes=(),
                              pin_topology=(), state_condition=(),
                              measurement_names=(), measurement_directions=(),
                              backend='ngspice', data_slew=0.015,
                              t_sim_end=3.0, temperature=80.0, supplies_hash='z')
    assert ctx._signature_load_key(sig_a) == ctx._signature_load_key(sig_b)


def test_worker_context_quarantine():
    """Quarantined signature key is stored."""
    from charlib.characterizer.reusable_engine import WorkerContext, TopologySignature
    ctx = WorkerContext()
    sig = TopologySignature(cell_hash='a', netlist_hash='b', model_hashes=(),
                            pin_topology=(), state_condition=(),
                            measurement_names=(), measurement_directions=(),
                            backend='ngspice', data_slew=0.015,
                            t_sim_end=3.0, temperature=25.0, supplies_hash='c')
    ctx.quarantine_signature(sig)
    key = ctx._signature_load_key(sig)
    assert key in ctx._quarantined_signatures


def test_worker_context_destruction():
    """Discard clears loaded state."""
    from charlib.characterizer.reusable_engine import WorkerContext
    ctx = WorkerContext()
    ctx.discard()
    assert not ctx.is_loaded



def test_worker_context_execute_reuses_loaded_signature():
    """execute_point_with_context reports one deck load per signature key."""
    from charlib.characterizer.reusable_engine import WorkerContext, TopologySignature

    class FakeShared:
        def __init__(self):
            self.loads = []
            self.commands = []
        def load_circuit(self, deck):
            self.loads.append(deck)
        def reset(self):
            self.commands.append('reset')
        def exec_command(self, command):
            self.commands.append(command)
        def run(self):
            return {'cell_fall__a_to_y': '1.0'}

    point = _make_point()
    sig = TopologySignature(cell_hash='a', netlist_hash='b', model_hashes=(),
                            pin_topology=(), state_condition=(),
                            measurement_names=('cell_fall__a_to_y',),
                            measurement_directions=(), backend='ngspice',
                            data_slew=point.data_slew, t_sim_end=point.t_sim_end,
                            temperature=point.temperature, supplies_hash='c')
    ctx = WorkerContext()
    fake = FakeShared()
    ctx._shared = fake

    first = ctx.execute_point_with_context(point, 'deck', sig, ('cell_fall__a_to_y',))
    second = ctx.execute_point_with_context(point, 'deck', sig, ('cell_fall__a_to_y',))

    assert first.status == 'ok'
    assert second.status == 'ok'
    assert first.deck_load_count == 1
    assert second.deck_load_count == 0
    assert ctx.deck_load_count == 1
    assert fake.loads == ['deck']
    assert 'alter cy c=2.0' in fake.commands


def test_worker_context_execute_quarantines_and_fresh_retries():
    """execute_point_with_context quarantines a failing signature and calls fresh retry once."""
    from charlib.characterizer.reusable_engine import WorkerContext, TopologySignature, MeasurementResult

    class FailingShared:
        def load_circuit(self, deck):
            pass
        def reset(self):
            pass
        def exec_command(self, command):
            raise RuntimeError('alter failed')

    point = _make_point()
    sig = TopologySignature(cell_hash='a', netlist_hash='b', model_hashes=(),
                            pin_topology=(), state_condition=(),
                            measurement_names=('cell_fall__a_to_y',),
                            measurement_directions=(), backend='ngspice',
                            data_slew=point.data_slew, t_sim_end=point.t_sim_end,
                            temperature=point.temperature, supplies_hash='c')
    ctx = WorkerContext()
    ctx._shared = FailingShared()
    calls = []
    def retry():
        calls.append(1)
        return MeasurementResult(point_id=point.point_id, task_id=point.task_id,
                                 measurement={'cell_fall__a_to_y': 1.0})

    result = ctx.execute_point_with_context(point, 'deck', sig, ('cell_fall__a_to_y',),
                                            fresh_retry=retry)

    assert result.status == 'ok'
    assert result.fresh_retry is True
    assert calls == [1]
    assert ctx._signature_load_key(sig) in ctx._quarantined_signatures


def test_worker_request_pickle_roundtrip():
    """WorkerRequest survives pickle round-trip."""
    import pickle
    from charlib.characterizer.reusable_engine import (
        SimulationPoint, TopologySignature, WorkerRequest
    )
    point = SimulationPoint(
        point_id='test', task_id=0, cell_name='INVX1',
        netlist_path='/tmp/test', model_paths=(), pin_order=('A', 'Y'),
        input_pin='A', output_pin='Y', input_transition='01', output_transition='10',
        stable_inputs=(), ignored_outputs=(), data_slew=0.015, load=0.06,
        t_sim_end=3.0, temperature=25.0, supplies=(), thresholds=(0.3, 0.7),
        time_unit_str='ns', voltage_unit_str='V', criterion_group='max',
        state_condition=()
    )
    sig = TopologySignature(
        cell_hash='a', netlist_hash='b', model_hashes=(),
        pin_topology=(), state_condition=(), measurement_names=(),
        measurement_directions=(), backend='ngspice',
        data_slew=0.015, t_sim_end=3.0, temperature=25.0, supplies_hash='c'
    )
    req = WorkerRequest(request_id='r1', point=point, deck_text='* test deck', signature=sig)
    data = pickle.dumps(req)
    req2 = pickle.loads(data)
    assert req2.request_id == 'r1'
    assert req2.point.point_id == 'test'
    assert req2.deck_text == '* test deck'


def test_worker_context_per_process_identity():
    """Each call to init_worker creates a fresh context with unique identity."""
    from charlib.characterizer.reusable_engine import (
        init_worker, get_worker_context, get_worker_id, shutdown_worker
    )
    init_worker(worker_id=1)
    ctx1 = get_worker_context()
    assert ctx1 is not None
    assert get_worker_id() == 1
    shutdown_worker()

    init_worker(worker_id=2)
    ctx2 = get_worker_context()
    assert ctx2 is not None
    assert ctx2 is not ctx1, "Different init_worker calls must create different contexts"
    assert get_worker_id() == 2
    shutdown_worker()
    assert get_worker_context() is None


def test_worker_result_no_pyspice_objects():
    """WorkerResult contains no PySpice or Liberty objects."""
    import pickle
    from charlib.characterizer.reusable_engine import WorkerResult, MeasurementResult
    r = WorkerResult(
        request_id='r1',
        result=MeasurementResult(point_id='p1', task_id=0, status='ok'),
        worker_id=0
    )
    data = pickle.dumps(r)
    r2 = pickle.loads(data)
    assert r2.request_id == 'r1'
    # Verify no PySpice/Liberty modules are referenced in the serialized data
    assert b'PySpice' not in data
    assert b'liberty' not in data.lower() or b'charlib' in data


def test_worker_context_create_destroy():
    """Context creation and destruction cycle is clean."""
    from charlib.characterizer.reusable_engine import (
        init_worker, get_worker_context, shutdown_worker
    )
    init_worker(worker_id=0)
    ctx = get_worker_context()
    assert ctx is not None
    assert ctx.deck_load_count == 0
    shutdown_worker()
    assert get_worker_context() is None
    # Re-initialize
    init_worker(worker_id=0)
    ctx2 = get_worker_context()
    assert ctx2 is not None
    assert ctx2 is not ctx
    shutdown_worker()


def test_scrambled_result_order():
    """LibertyAssembler sorting by point_id restores determinism."""
    from charlib.characterizer.reusable_engine import LibertyAssembler, MeasurementResult
    assembler = LibertyAssembler()
    results = [
        MeasurementResult(point_id='point_z', task_id=0),
        MeasurementResult(point_id='point_a', task_id=1),
        MeasurementResult(point_id='point_m', task_id=2),
    ]
    assembler.add_results(results)
    ordered = assembler.ordered_results()
    assert [result.point_id for result in ordered] == ['point_a', 'point_m', 'point_z']


def test_worker_failure_graceful():
    """A None point produces an error WorkerResult, not an exception."""
    from charlib.characterizer.reusable_engine import execute_request, WorkerRequest
    req = WorkerRequest(request_id='r1', point=None, deck_text='', signature=None)
    result = execute_request(req)
    assert result.request_id == 'r1'
    assert result.result.status == 'error'
    assert result.worker_error is not None


def test_missing_point_detection():
    """detect_missing_requests reports requests that have no matching result."""
    from charlib.characterizer.reusable_engine import (
        detect_missing_requests, WorkerRequest, WorkerResult, MeasurementResult
    )
    requests = [
        WorkerRequest(request_id='a', point=None, deck_text='', signature=None),
        WorkerRequest(request_id='b', point=None, deck_text='', signature=None),
    ]
    results = [WorkerResult(request_id='a',
                            result=MeasurementResult(point_id='a', task_id=0))]
    assert detect_missing_requests(requests, results) == ['b']


# ---------------------------------------------------------------------------
# Batch protocol tests
# ---------------------------------------------------------------------------

def _make_signature(name="sig"):
    return TopologySignature(
        cell_hash=name,
        netlist_hash='b',
        model_hashes=(),
        pin_topology=(),
        state_condition=(),
        measurement_names=('cell_fall__a_to_y',),
        measurement_directions=(),
        backend='ngspice',
        data_slew=0.015,
        t_sim_end=3.0,
        temperature=25.0,
        supplies_hash='c',
    )


class _FakeShared:
    """Minimal stand-in for PySpice shared ngspice object."""
    def __init__(self, measurements=None, fail_on_run_for=None):
        self.loads = []
        self.commands = []
        self.measurements = measurements or {'cell_fall__a_to_y': '1.0'}
        self.fail_on_run_for = set(fail_on_run_for or [])

    def load_circuit(self, deck):
        self.loads.append(deck)

    def reset(self):
        self.commands.append('reset')

    def exec_command(self, command):
        self.commands.append(command)

    def run(self):
        # Use exec_command history to identify which point is running. This is a
        # crude but deterministic proxy: the last alter command tells us the
        # load value currently being simulated.
        for cmd in reversed(self.commands):
            if cmd.startswith('alter') and 'c=99.0' in cmd:
                if '99.0' in self.fail_on_run_for:
                    raise RuntimeError('simulated run failure')
        return self.measurements


def _init_fake_worker(shared):
    from charlib.characterizer.reusable_engine import init_worker, get_worker_context, shutdown_worker
    shutdown_worker()
    init_worker(worker_id=42)
    ctx = get_worker_context()
    ctx._shared = shared
    return ctx


def test_batch_types_pickleable():
    """SignatureGroup, WorkerBatchRequest, and WorkerBatchResult are pickleable."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, WorkerBatchRequest, WorkerBatchResult, WorkerResult
    )
    point = _make_point()
    sig = _make_signature()
    group = SignatureGroup(
        signature=sig,
        deck_text='deck',
        measurement_names=('cell_fall__a_to_y',),
        sweep_points=(point,),
    )
    req = WorkerBatchRequest(batch_id='batch_0', groups=(group,))
    data = pickle.dumps(req)
    req2 = pickle.loads(data)
    assert req2.batch_id == 'batch_0'
    assert req2.groups[0].deck_text == 'deck'
    assert req2.groups[0].sweep_points[0].point_id == point.point_id

    result = WorkerBatchResult(
        batch_id='batch_0',
        results=(WorkerResult(request_id='p1', result=MeasurementResult(point_id='p1', task_id=1)),),
        worker_pid=1234,
    )
    data = pickle.dumps(result)
    result2 = pickle.loads(data)
    assert result2.batch_id == 'batch_0'
    assert result2.results[0].request_id == 'p1'
    assert result2.worker_pid == 1234


def test_execute_batch_one_load_per_signature():
    """execute_batch loads each signature exactly once."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, WorkerBatchRequest, execute_batch
    )
    shared = _FakeShared()
    _init_fake_worker(shared)

    sig_a = _make_signature('a')
    sig_b = _make_signature('b')
    points_a = [_make_point('a1', load=1.0), _make_point('a2', load=2.0)]
    points_b = [_make_point('b1', load=3.0)]
    request = WorkerBatchRequest(
        batch_id='batch',
        groups=(
            SignatureGroup(signature=sig_a, deck_text='deck_a',
                           measurement_names=('cell_fall__a_to_y',), sweep_points=tuple(points_a)),
            SignatureGroup(signature=sig_b, deck_text='deck_b',
                           measurement_names=('cell_fall__a_to_y',), sweep_points=tuple(points_b)),
        ),
    )
    result = execute_batch(request)
    assert result.batch_id == 'batch'
    assert len(result.results) == 3
    # Two distinct decks loaded, once each.
    assert shared.loads == ['deck_a', 'deck_b']
    # Each point produced an ok result.
    assert all(r.result.status == 'ok' for r in result.results)


def test_execute_batch_compact_results():
    """execute_batch returns only the requested measurement names."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, WorkerBatchRequest, execute_batch
    )
    shared = _FakeShared(measurements={
        'cell_fall__a_to_y': '1.0',
        'cell_rise__a_to_y': '2.0',
        'unwanted_metric': '3.0',
    })
    _init_fake_worker(shared)

    point = _make_point('p1')
    sig = _make_signature()
    request = WorkerBatchRequest(
        batch_id='batch',
        groups=(SignatureGroup(signature=sig, deck_text='deck',
                                measurement_names=('cell_fall__a_to_y', 'cell_rise__a_to_y'),
                                sweep_points=(point,)),),
    )
    result = execute_batch(request)
    assert len(result.results) == 1
    assert result.results[0].result.measurement == {
        'cell_fall__a_to_y': 1.0,
        'cell_rise__a_to_y': 2.0,
    }
    assert 'unwanted_metric' not in result.results[0].result.measurement


def test_execute_batch_worker_crash_graceful():
    """execute_batch returns an error result, not an exception, when run() fails."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, WorkerBatchRequest, execute_batch
    )

    class CrashingShared:
        def load_circuit(self, deck):
            pass
        def reset(self):
            pass
        def exec_command(self, command):
            pass
        def run(self):
            raise RuntimeError('ngspice crashed')

    _init_fake_worker(CrashingShared())
    point = _make_point('p1')
    sig = _make_signature()
    request = WorkerBatchRequest(
        batch_id='batch',
        groups=(SignatureGroup(signature=sig, deck_text='deck',
                                measurement_names=('cell_fall__a_to_y',),
                                sweep_points=(point,)),),
    )
    result = execute_batch(request)
    assert len(result.results) == 1
    assert result.results[0].result.status == 'error'
    assert 'ngspice crashed' in result.results[0].result.error


def test_batch_result_no_pyspice_objects():
    """WorkerBatchResult pickle stream contains no PySpice/Liberty references."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, WorkerBatchRequest, WorkerBatchResult, WorkerResult
    )
    point = _make_point()
    group = SignatureGroup(
        signature=_make_signature(), deck_text='deck',
        measurement_names=('cell_fall__a_to_y',), sweep_points=(point,),
    )
    req = WorkerBatchRequest(batch_id='batch_0', groups=(group,))
    data = pickle.dumps(req)
    assert b'PySpice' not in data
    assert b'NgSpiceShared' not in data

    result = WorkerBatchResult(
        batch_id='batch_0',
        results=(WorkerResult(request_id='p1', result=MeasurementResult(point_id='p1', task_id=1)),),
        worker_pid=1234,
    )
    data = pickle.dumps(result)
    assert b'PySpice' not in data
    assert b'NgSpiceShared' not in data


def test_lpt_batch_groups_sorts_and_balances():
    """lpt_batch_groups sorts groups descending and balances point counts."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, lpt_batch_groups
    )
    sig = _make_signature()
    groups = [
        SignatureGroup(signature=sig, deck_text=f'deck_{i}',
                       measurement_names=(), sweep_points=tuple(_make_point(f'p{i}_{j}') for j in range(size)))
        for i, size in enumerate([5, 3, 2, 1])
    ]
    batches = lpt_batch_groups(groups, num_workers=1)
    # With 4 groups and 1 worker, max_batches = 4, so each group is its own batch.
    assert len(batches) == 4
    # LPT order: largest group first.
    assert [len(b[0].sweep_points) for b in batches] == [5, 3, 2, 1]


def test_lpt_batch_groups_bounded_in_flight():
    """lpt_batch_groups never produces more than 4*N batches."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, lpt_batch_groups
    )
    sig = _make_signature()
    groups = [
        SignatureGroup(signature=sig, deck_text=f'deck_{i}',
                       measurement_names=(), sweep_points=tuple(_make_point(f'p{i}_{j}') for j in range(2)))
        for i in range(20)
    ]
    for num_workers in (1, 2, 4, 8):
        batches = lpt_batch_groups(groups, num_workers=num_workers)
        assert len(batches) <= 4 * num_workers
        # All groups are preserved
        assert sum(len(b) for b in batches) == len(groups)


def test_lpt_batch_groups_same_signature_not_split():
    """A single SignatureGroup is never split across batches."""
    from charlib.characterizer.reusable_engine import (
        SignatureGroup, lpt_batch_groups
    )
    sig = _make_signature()
    group = SignatureGroup(
        signature=sig,
        deck_text='deck',
        measurement_names=(),
        sweep_points=tuple(_make_point(f'p{i}') for i in range(10)),
    )
    batches = lpt_batch_groups([group], num_workers=4)
    assert len(batches) == 1
    assert batches[0][0] is group


def test_jobs_missing_uses_multithreaded_default():
    """When jobs is not in kwargs, multithreaded default (None) is used."""
    from charlib.characterizer.characterizer import CharacterizationSettings
    s = CharacterizationSettings()
    assert s.jobs is None, f"Expected None (multithreaded default), got {s.jobs}"


def test_jobs_explicit_value_passed_through():
    """Explicit jobs=4 reaches CharacterizationSettings.jobs."""
    from charlib.characterizer.characterizer import CharacterizationSettings
    for val in [1, 4, 8, None]:
        s = CharacterizationSettings(jobs=val)
        assert s.jobs == val, f"Expected jobs={val}, got {s.jobs}"


def test_jobs_explicit_zero_not_silently_dropped():
    """Explicit jobs=0 is preserved, not silently replaced by multithreaded."""
    from charlib.characterizer.characterizer import CharacterizationSettings
    s = CharacterizationSettings(jobs=0)
    assert s.jobs == 0, "jobs=0 must not be silently replaced"


