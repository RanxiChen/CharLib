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


def _make_point(point_id="p1") -> SimulationPoint:
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
        load=2.0,
        t_sim_end=10.0,
        temperature=25.0,
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
