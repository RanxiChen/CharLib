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
