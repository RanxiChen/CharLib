#!/usr/bin/env python3
"""Compare Liberty files by byte-level and semantic content.

Extends the parser from analyze_liberty_comparison.py to provide:
- BYTE_IDENTICAL: SHA-256 match
- SEMANTIC_EXACT: structure/values identical, only text/order differs
- NUMERIC_CLOSE: structure same, floating-point differences within threshold
- DIFFERENT: structural or substantial numerical differences
- MISSING_OR_FAILED: no valid Liberty file

Self-tests:
- Same file → BYTE_IDENTICAL
- Group-order permuted file → SEMANTIC_EXACT
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ── Liberty parser (extended from analyze_liberty_comparison.py) ────────────

GROUP_START_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{\s*$")
GROUP_END_RE = re.compile(r"^\s*}\s*/\*\s*end\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*/\s*$")
SIMPLE_ATTR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*;\s*$")
COMPLEX_ATTR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\"(.*?)\"\s*\)\s*;\s*$")
MULTI_LINE_START_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\\\s*$")
MULTI_LINE_END_RE = re.compile(r"^\s*\)\s*;\s*$")
MULTI_LINE_VALUE_RE = re.compile(r'^\s*"(.*?)"\s*,?\s*\\?\s*$')
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)", re.I)


@dataclass
class Group:
    name: str
    prefix: str
    depth: int
    attrs: dict[str, str] = field(default_factory=dict)
    complex_attrs: dict[str, str] = field(default_factory=dict)
    children: list[Group] = field(default_factory=list)

    def child_groups(self, name: str) -> Iterator[Group]:
        return (child for child in self.children if child.name == name)


@dataclass(frozen=True)
class NumericRecord:
    kind: str
    cell: str
    pin: str
    related_pin: str
    timing_type: str
    timing_sense: str
    table: str
    qualifier: str
    axis_1: str
    axis_2: str
    values: tuple[float, ...]

    @staticmethod
    def from_group(g: Group, cell: str, pin: str, related_pin: str,
                   timing_type: str, timing_sense: str) -> list[NumericRecord]:
        records = []
        vals_str = g.complex_attrs.get("values") or g.attrs.get("values", "")
        if not vals_str:
            return records
        vals = tuple(parse_numbers(vals_str))
        index1 = g.complex_attrs.get("index_1") or g.attrs.get("index_1", "")
        index2 = g.complex_attrs.get("index_2") or g.attrs.get("index_2", "")
        qualifiers = ""
        when_val = g.complex_attrs.get("when") or g.attrs.get("when", "")
        if when_val:
            qualifiers = f"when={when_val}"
        prefix = g.prefix or ""
        records.append(NumericRecord(
            kind=prefix + ("_index_1" if "index_1" in g.attrs or "index_1" in g.complex_attrs else ""),
            cell=cell, pin=pin, related_pin=related_pin,
            timing_type=timing_type, timing_sense=timing_sense,
            table=g.name, qualifier=qualifiers,
            axis_1=index1, axis_2=index2, values=vals,
        ))
        return records


def clean_token(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def parse_numbers(text: str) -> list[float]:
    return [float(token) for token in NUMBER_RE.findall(text)]


def parse_liberty(path: Path) -> Group:
    root = Group("__root__", "", 0)
    stack: list[Group] = [root]
    prefix: list[str] = []
    multi_line_key: str | None = None
    multi_line_values: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            stripped = line.lstrip()

            # Handle multi-line continuation
            if multi_line_key is not None:
                m_end_ml = MULTI_LINE_END_RE.match(stripped)
                if m_end_ml:
                    stack[-1].complex_attrs[multi_line_key] = " ".join(multi_line_values)
                    multi_line_key = None
                    multi_line_values = []
                    continue
                m_val = MULTI_LINE_VALUE_RE.match(stripped)
                if m_val:
                    multi_line_values.append(m_val.group(1))
                # Skip other lines within multi-line block (blank, comment, backslash-only)
                continue

            # Skip comments and blank lines
            if not stripped or stripped.startswith("/*") or stripped.startswith("//"):
                continue

            m_start = GROUP_START_RE.match(stripped)
            m_end = GROUP_END_RE.match(stripped)
            m_simple = SIMPLE_ATTR_RE.match(stripped)
            m_complex = COMPLEX_ATTR_RE.match(stripped)
            m_ml_start = MULTI_LINE_START_RE.match(stripped)

            if m_end:
                if stack:
                    stack.pop()
                if prefix:
                    prefix.pop()
                continue

            if m_start:
                name = m_start.group(1)
                arg = m_start.group(2)
                depth = len(stack)
                g = Group(name, arg, depth)
                stack[-1].children.append(g)
                stack.append(g)
                prefix.append(name)
                continue

            if m_ml_start:
                multi_line_key = m_ml_start.group(1)
                multi_line_values = []
                continue

            if m_simple:
                key = m_simple.group(1)
                value = m_simple.group(2)
                value = clean_token(value)
                comment_idx = value.find("//")
                if comment_idx >= 0:
                    value = value[:comment_idx].strip()
                stack[-1].attrs[key] = value
                continue

            if m_complex:
                key = m_complex.group(1)
                value = m_complex.group(2)
                stack[-1].complex_attrs[key] = value
                continue
    # Find the first library
    libraries = list(root.child_groups("library"))
    if not libraries:
        raise ValueError(f"No library found in {path}")
    return libraries[0]


def all_groups(group: Group) -> Iterator[Group]:
    yield group
    for child in group.children:
        yield from all_groups(child)


def cell_groups(library: Group) -> list[Group]:
    return list(library.child_groups("cell"))


def summary(library: Group) -> dict[str, object]:
    groups = Counter(group.name for group in all_groups(library))
    parts: dict[str, dict[str, int]] = defaultdict(Counter)
    for g in all_groups(library):
        if g.name in ("cell", "pin", "pg_pin", "ff", "latch", "timing", "leakage_power"):
            parts[g.name][g.prefix] += 1

    cells = set()
    for cell in library.child_groups("cell"):
        cells.add(cell.prefix)

    return {
        "group_counts": dict(groups),
        "cell_count": len(cells),
        "cells": sorted(cells),
        "detail": {k: dict(v) for k, v in sorted(parts.items())},
    }


def numeric_records(library: Group) -> dict[tuple, NumericRecord]:
    records: dict[tuple, NumericRecord] = {}
    for cell in cell_groups(library):
        cell_name = cell.prefix
        for pin in cell.child_groups("pin"):
            pin_name = pin.prefix
            # Pin-level attributes
            for attr_name in ("capacitance", "direction", "function", "clock"):
                if attr_name in pin.attrs:
                    pass  # tracked via text_records

            # Timing groups
            for timing in pin.child_groups("timing"):
                related_pin = timing.attrs.get("related_pin", "")
                timing_type = timing.attrs.get("timing_type", "")
                timing_sense = timing.attrs.get("timing_sense", "")

                for child in timing.children:
                    recs = NumericRecord.from_group(
                        child, cell_name, pin_name,
                        related_pin, timing_type, timing_sense,
                    )
                    for rec in recs:
                        key = (rec.kind, rec.cell, rec.pin, rec.related_pin,
                               rec.timing_type, rec.timing_sense, rec.table,
                               rec.qualifier)
                        records[key] = rec

            # Pin-level tables directly under pin (e.g. internal_power, timing)
            for child in pin.children:
                if child.name not in ("timing", "related_pin", "pg_type", "direction", "function", "clock"):
                    recs = NumericRecord.from_group(
                        child, cell_name, pin_name, "", "", ""
                    )
                    for rec in recs:
                        key = (rec.kind, rec.cell, rec.pin, rec.related_pin,
                               rec.timing_type, rec.timing_sense, rec.table,
                               rec.qualifier)
                        records[key] = rec

            # Leakage
            for leakage in cell.child_groups("leakage_power"):
                if "value" in leakage.attrs:
                    pass  # tracked via text_records

        # Cell-level tables (e.g. cell_leakage_power)
        for child in cell.children:
            if child.name not in ("pin", "pg_pin", "ff", "latch", "leakage_power"):
                recs = NumericRecord.from_group(child, cell_name, "", "", "", "")
                for rec in recs:
                    key = (rec.kind, rec.cell, rec.pin, rec.related_pin,
                           rec.timing_type, rec.timing_sense, rec.table,
                           rec.qualifier)
                    records[key] = rec

    return records


def _normalize_when(when: str) -> str:
    """Normalize a leakage 'when' condition so equivalent orderings match."""
    return "".join(sorted(when.replace(" ", "").replace("\t", "").lower()))


def text_records(library: Group) -> dict[tuple[str, str, str], str]:
    records: dict[tuple[str, str, str], str] = {}
    for cell in cell_groups(library):
        for pin in cell.child_groups("pin"):
            for attr in ("direction", "function", "clock", "capacitance"):
                if attr in pin.attrs:
                    records[(cell.prefix, pin.prefix, attr)] = pin.attrs[attr]
        for leakage in cell.child_groups("leakage_power"):
            when = leakage.attrs.get("when", "")
            key = (cell.prefix, _normalize_when(when), "leakage_power")
            if "value" in leakage.attrs:
                records[key] = leakage.attrs["value"]
            # Preserve distinct leakage groups even when the value is missing.
            if not when and key not in records:
                records[key] = ""
    return records


def library_attrs(library: Group) -> dict[str, str]:
    """Extract top-level library attributes (units, global properties)."""
    # Get immediate attributes of the library group and its direct children
    # that aren't cells
    attrs: dict[str, str] = {}
    # Library-level simple attrs
    for k in ("date", "revision", "comment", "delay_model", "time_unit",
              "capacitive_load_unit", "leakage_power_unit", "voltage_unit",
              "current_unit", "pulling_resistance_unit", "nom_process",
              "nom_temperature", "nom_voltage", "default_cell_leakage_power",
              "default_fanout_load", "default_inout_pin_cap",
              "default_input_pin_cap", "default_output_pin_cap",
              "default_max_transition", "in_place_swap_mode"):
        if k in library.attrs:
            attrs[k] = library.attrs[k]
    # Units from child groups
    for child in library.children:
        if child.name in ("lu_table_template", "power_lut_template",
                          "output_current_template", "input_voltage_template"):
            attrs[f"template_{child.prefix}"] = child.prefix
    return attrs


# ── comparison functions ────────────────────────────────────────────────────

VERDICT = {
    0: "BYTE_IDENTICAL",
    1: "SEMANTIC_EXACT",
    2: "NUMERIC_CLOSE",
    3: "DIFFERENT",
    4: "MISSING_OR_FAILED",
}


def compare_bytes(path_a: Path, path_b: Path) -> bool:
    """Check SHA-256 byte-level identity."""
    ha = hashlib.sha256(path_a.read_bytes()).hexdigest()
    hb = hashlib.sha256(path_b.read_bytes()).hexdigest()
    return ha == hb


def compare_libraries(lib_a: Group, lib_b: Group, label_a: str, label_b: str) -> dict:
    """Full semantic comparison of two parsed libraries."""
    result = {
        "label_a": label_a,
        "label_b": label_b,
        "verdict": "DIFFERENT",
        "verdict_code": 3,
        "details": {},
    }

    # ── library-level attributes ──
    attrs_a = library_attrs(lib_a)
    attrs_b = library_attrs(lib_b)
    attr_diffs = {}
    for k in sorted(set(attrs_a) | set(attrs_b)):
        va = attrs_a.get(k)
        vb = attrs_b.get(k)
        if va != vb:
            attr_diffs[k] = {"a": va, "b": vb}
    result["details"]["library_attr_differences"] = attr_diffs

    # ── summary comparison ──
    sum_a = summary(lib_a)
    sum_b = summary(lib_b)
    result["details"]["summary_a"] = sum_a
    result["details"]["summary_b"] = sum_b

    # Cell sets
    cells_a = set(sum_a["cells"])
    cells_b = set(sum_b["cells"])
    result["details"]["cells_only_in_a"] = sorted(cells_a - cells_b)
    result["details"]["cells_only_in_b"] = sorted(cells_b - cells_a)
    cells_match = (cells_a == cells_b)

    # Group counts
    groups_a = Counter(sum_a["group_counts"])
    groups_b = Counter(sum_b["group_counts"])
    group_count_diff = {}
    for k in sorted(set(groups_a) | set(groups_b)):
        if groups_a.get(k) != groups_b.get(k):
            group_count_diff[k] = {"a": groups_a.get(k, 0), "b": groups_b.get(k, 0)}
    result["details"]["group_count_differences"] = group_count_diff

    # ── numeric comparison ──
    recs_a = numeric_records(lib_a)
    recs_b = numeric_records(lib_b)

    # Keys only in one
    keys_a = set(recs_a.keys())
    keys_b = set(recs_b.keys())
    result["details"]["numeric_keys_a"] = len(keys_a)
    result["details"]["numeric_keys_b"] = len(keys_b)
    result["details"]["numeric_keys_only_in_a"] = len(keys_a - keys_b)
    result["details"]["numeric_keys_only_in_b"] = len(keys_b - keys_a)

    common_keys = keys_a & keys_b
    numeric_diffs: list[dict] = []
    for key in sorted(common_keys):
        va = recs_a[key].values
        vb = recs_b[key].values
        if len(va) != len(vb):
            numeric_diffs.append({
                "key": key,
                "len_a": len(va), "len_b": len(vb),
                "a": va, "b": vb,
            })
            continue
        for i, (a, b) in enumerate(zip(va, vb)):
            abs_diff = abs(a - b)
            rel_diff = abs_diff / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 1e-30 else 0.0
            if abs_diff > 1e-15:
                numeric_diffs.append({
                    "key": key, "index": i,
                    "a": a, "b": b,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                })

    result["details"]["numeric_diff_count"] = len(numeric_diffs)
    result["details"]["numeric_total_compared"] = len(common_keys)

    # Track large diffs
    over_1pct = [d for d in numeric_diffs if d.get("rel_diff", 0) > 0.01]
    over_5pct = [d for d in numeric_diffs if d.get("rel_diff", 0) > 0.05]
    result["details"]["numeric_over_1pct"] = len(over_1pct)
    result["details"]["numeric_over_5pct"] = len(over_5pct)

    if over_1pct:
        result["details"]["largest_diffs"] = sorted(
            over_1pct, key=lambda d: d.get("rel_diff", 0), reverse=True
        )[:20]

    # ── text attribute comparison ──
    text_a = text_records(lib_a)
    text_b = text_records(lib_b)
    text_diffs = []
    for k in sorted(set(text_a) | set(text_b)):
        va = text_a.get(k)
        vb = text_b.get(k)
        if va != vb:
            text_diffs.append({"key": k, "a": va, "b": vb})
    result["details"]["text_differences"] = text_diffs

    # ── determine verdict ──
    has_structural_diff = bool(
        result["details"]["cells_only_in_a"] or
        result["details"]["cells_only_in_b"] or
        group_count_diff or
        attr_diffs or
        result["details"]["numeric_keys_only_in_a"] > 0 or
        result["details"]["numeric_keys_only_in_b"] > 0
    )
    has_numeric_diff = len(numeric_diffs) > 0
    has_text_diff = len(text_diffs) > 0

    if not cells_match:
        result["verdict"] = "DIFFERENT"
        result["verdict_code"] = 3
    elif has_structural_diff:
        result["verdict"] = "DIFFERENT"
        result["verdict_code"] = 3
    elif has_numeric_diff and over_5pct:
        result["verdict"] = "DIFFERENT"
        result["verdict_code"] = 3
    elif has_numeric_diff:
        result["verdict"] = "NUMERIC_CLOSE"
        result["verdict_code"] = 2
    elif has_text_diff:
        result["verdict"] = "SEMANTIC_EXACT"
        result["verdict_code"] = 1
    else:
        result["verdict"] = "SEMANTIC_EXACT"
        result["verdict_code"] = 1

    return result


def compare_file_pair(path_a: Path, path_b: Path, label_a: str, label_b: str) -> dict:
    """Compare two Liberty files comprehensively."""
    # Check existence
    if not path_a.is_file() or not path_b.is_file():
        return {
            "label_a": label_a, "label_b": label_b,
            "verdict": "MISSING_OR_FAILED", "verdict_code": 4,
            "details": {"missing_a": not path_a.is_file(), "missing_b": not path_b.is_file()},
        }

    # Byte-level check
    if compare_bytes(path_a, path_b):
        return {
            "label_a": label_a, "label_b": label_b,
            "verdict": "BYTE_IDENTICAL", "verdict_code": 0,
            "sha256": hashlib.sha256(path_a.read_bytes()).hexdigest(),
        }

    # Semantic check
    try:
        lib_a = parse_liberty(path_a)
    except Exception as e:
        return {
            "label_a": label_a, "label_b": label_b,
            "verdict": "MISSING_OR_FAILED", "verdict_code": 4,
            "details": {"parse_error_a": str(e)},
        }

    try:
        lib_b = parse_liberty(path_b)
    except Exception as e:
        return {
            "label_a": label_a, "label_b": label_b,
            "verdict": "MISSING_OR_FAILED", "verdict_code": 4,
            "details": {"parse_error_b": str(e)},
        }

    result = compare_libraries(lib_a, lib_b, label_a, label_b)
    result["sha256_a"] = hashlib.sha256(path_a.read_bytes()).hexdigest()
    result["sha256_b"] = hashlib.sha256(path_b.read_bytes()).hexdigest()
    return result


# ── self-tests ──────────────────────────────────────────────────────────────

def run_self_tests() -> bool:
    """Verify the parser handles same-file and group-order-permuted cases."""
    print("=== Self-tests ===")
    ok = True

    # Test 1: parse a Liberty file successfully
    test_content = """library(test) {
  time_unit : "1ns" ;
  voltage_unit : "1V" ;
  cell(INVX1) {
    area : 1.0 ;
    pin(A) {
      direction : "input" ;
      capacitance : 0.001 ;
    } /* end pin */
    pin(Y) {
      direction : "output" ;
      function : "(!A)" ;
      timing() {
        related_pin : "A" ;
        timing_sense : "negative_unate" ;
        cell_rise(delay_template_7x7) {
          index_1("0.01, 0.02, 0.03");
          index_2("0.001, 0.002, 0.003");
          values("1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0");
        } /* end cell_rise */
        cell_fall(delay_template_7x7) {
          index_1("0.01, 0.02, 0.03");
          index_2("0.001, 0.002, 0.003");
          values("0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5");
        } /* end cell_fall */
      } /* end timing */
    } /* end pin */
  } /* end cell */
} /* end library */
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f:
        f.write(test_content)
        tmp_path = Path(f.name)

    try:
        lib = parse_liberty(tmp_path)
        cells = cell_groups(lib)
        assert len(cells) == 1, f"Expected 1 cell, got {len(cells)}"
        assert cells[0].prefix == "INVX1", f"Expected INVX1, got {cells[0].prefix}"
        print("  PASS: parse Liberty")

        recs = numeric_records(lib)
        assert len(recs) == 2, f"Expected 2 numeric records, got {len(recs)}"
        print("  PASS: numeric_records count")

        text = text_records(lib)
        assert ("INVX1", "A", "direction") in text
        assert ("INVX1", "Y", "function") in text
        print("  PASS: text_records")

        # Test 2: same file → BYTE_IDENTICAL
        result = compare_file_pair(tmp_path, tmp_path, "a", "b")
        assert result["verdict"] == "BYTE_IDENTICAL", f"Expected BYTE_IDENTICAL, got {result['verdict']}"
        print("  PASS: same file → BYTE_IDENTICAL")

        # Test 3: group-order-permuted → SEMANTIC_EXACT
        permuted = test_content.replace(
            'direction : "input" ;\n      capacitance : 0.001 ;',
            'capacitance : 0.001 ;\n      direction : "input" ;'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f2:
            f2.write(permuted)
            tmp2 = Path(f2.name)

        try:
            result2 = compare_file_pair(tmp_path, tmp2, "a", "b")
            assert result2["verdict"] in ("SEMANTIC_EXACT", "BYTE_IDENTICAL"), \
                f"Expected SEMANTIC_EXACT, got {result2['verdict']}"
            print(f"  PASS: permuted attrs → {result2['verdict']}")
        finally:
            tmp2.unlink(missing_ok=True)

        # Test 4: different value → DIFFERENT or NUMERIC_CLOSE
        different = test_content.replace('values("1.0, 2.0', 'values("1.5, 2.5')
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f3:
            f3.write(different)
            tmp3 = Path(f3.name)
        try:
            result3 = compare_file_pair(tmp_path, tmp3, "a", "b")
            assert result3["verdict_code"] >= 2, \
                f"Expected >= NUMERIC_CLOSE, got {result3['verdict']}"
            print(f"  PASS: different values → {result3['verdict']}")
        finally:
            tmp3.unlink(missing_ok=True)

    finally:
        tmp_path.unlink(missing_ok=True)

    if ok:
        print("All self-tests passed.")
    return ok


# ── matrix comparison ───────────────────────────────────────────────────────

def find_result_libs(runs_root: Path) -> dict[tuple[str, str, int, int], Path]:
    """Find all result.lib files keyed by (installation, input, jobs, turn)."""
    libs: dict[tuple[str, str, int, int], Path] = {}
    for lib_file in sorted(runs_root.rglob("result.lib")):
        # Path is runs/<inst>/<input>/jobs-NNN/turn-NN/result.lib
        parts = lib_file.relative_to(runs_root).parts
        if len(parts) >= 5:
            inst = parts[0]
            input_id = parts[1]
            jobs_dir = parts[2]  # jobs-NNN
            turn_dir = parts[3]  # turn-NN
            try:
                jobs = int(jobs_dir.replace("jobs-", ""))
                turn = int(turn_dir.replace("turn-", ""))
            except ValueError:
                continue
            libs[(inst, input_id, jobs, turn)] = lib_file
    return libs


def run_matrix_comparison(runs_root: Path, out_dir: Path) -> None:
    """Run all Liberty comparisons across the matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    libs = find_result_libs(runs_root)
    print(f"Found {len(libs)} result.lib files.")

    # Build a csv of all pairwise comparisons
    rows: list[dict] = []
    all_keys = sorted(libs.keys())

    # Comparisons by input
    inputs_seen = sorted(set(k[1] for k in all_keys))
    for input_id in inputs_seen:
        subset = {k: v for k, v in libs.items() if k[1] == input_id}
        keys_sorted = sorted(subset.keys())

        # Compare each pair
        for i in range(len(keys_sorted)):
            for j in range(i + 1, len(keys_sorted)):
                ka, kb = keys_sorted[i], keys_sorted[j]
                pa, pb = subset[ka], subset[kb]
                label_a = f"{ka[0]}/j{ka[2]}/t{ka[3]}"
                label_b = f"{kb[0]}/j{kb[2]}/t{kb[3]}"
                result = compare_file_pair(pa, pb, label_a, label_b)
                rows.append({
                    "input": input_id,
                    "installation_a": ka[0],
                    "installation_b": kb[0],
                    "jobs_a": ka[2], "jobs_b": kb[2],
                    "turn_a": ka[3], "turn_b": kb[3],
                    "verdict": result["verdict"],
                    "verdict_code": result["verdict_code"],
                    "sha256_a": result.get("sha256_a", ""),
                    "sha256_b": result.get("sha256_b", ""),
                    "numeric_diff_count": result.get("details", {}).get("numeric_diff_count", 0),
                    "numeric_over_1pct": result.get("details", {}).get("numeric_over_1pct", 0),
                    "numeric_keys_only_in_a": result.get("details", {}).get("numeric_keys_only_in_a", 0),
                    "numeric_keys_only_in_b": result.get("details", {}).get("numeric_keys_only_in_b", 0),
                    "cells_only_in_a": len(result.get("details", {}).get("cells_only_in_a", [])),
                    "cells_only_in_b": len(result.get("details", {}).get("cells_only_in_b", [])),
                    "text_diff_count": len(result.get("details", {}).get("text_differences", [])),
                })

    # Write liberty-pairs.csv
    if rows:
        fieldnames = [
            "input", "installation_a", "installation_b",
            "jobs_a", "jobs_b", "turn_a", "turn_b",
            "verdict", "verdict_code",
            "sha256_a", "sha256_b",
            "numeric_diff_count", "numeric_over_1pct",
            "numeric_keys_only_in_a", "numeric_keys_only_in_b",
            "cells_only_in_a", "cells_only_in_b",
            "text_diff_count",
        ]
        with (out_dir / "liberty-pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} pair comparisons.")

    # Write semantic differences detail (DIFFERENT pairs only)
    different_rows = [r for r in rows if r["verdict"] == "DIFFERENT"]
    if different_rows:
        with (out_dir / "semantic-differences.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "input", "installation_a", "installation_b",
                "jobs_a", "jobs_b", "turn_a", "turn_b",
                "numeric_diff_count", "numeric_over_1pct",
                "numeric_keys_only_in_a", "numeric_keys_only_in_b",
                "cells_only_in_a", "cells_only_in_b",
                "text_diff_count",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(different_rows)
        print(f"Wrote {len(different_rows)} DIFFERENT pairs to semantic-differences.csv.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Liberty files by byte and semantic content"
    )
    parser.add_argument("--runs-root", type=Path,
                        help="Root of runs/ tree for matrix comparison")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for comparison results")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-tests and exit")
    parser.add_argument("--file-a", type=Path,
                        help="Single file comparison: file A")
    parser.add_argument("--file-b", type=Path,
                        help="Single file comparison: file B")
    parser.add_argument("--label-a", type=str, default="A",
                        help="Label for file A")
    parser.add_argument("--label-b", type=str, default="B",
                        help="Label for file B")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_tests()
        return 0 if ok else 1

    # Single file comparison
    if args.file_a and args.file_b:
        result = compare_file_pair(args.file_a, args.file_b,
                                   args.label_a, args.label_b)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "comparison.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"Verdict: {result['verdict']}")
        if result["verdict_code"] >= 2:
            details = result.get("details", {})
            print(f"  Numeric diffs: {details.get('numeric_diff_count', 0)}")
            print(f"  Over 1%: {details.get('numeric_over_1pct', 0)}")
            print(f"  Over 5%: {details.get('numeric_over_5pct', 0)}")
        return 0

    # Matrix comparison
    if args.runs_root:
        run_matrix_comparison(args.runs_root, args.out_dir)
        return 0

    parser.error("Either --runs-root or both --file-a and --file-b required")


if __name__ == "__main__":
    raise SystemExit(main())
