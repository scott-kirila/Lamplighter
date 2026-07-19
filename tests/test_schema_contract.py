"""Drift guard for the Python↔TypeScript wire contract.

The frontend's ``Domain*`` interfaces (frontend/src/types/graph.ts) are a
hand-maintained mirror of the pydantic models in ``backend/schema.py`` — load/save
is a straight JSON pass-through, so their field names must match exactly. This test
fails when a backend field is renamed/added/removed without updating the mirror
(and vice versa), turning silent drift into a red build with a pointer to the fix.
"""
import re
from pathlib import Path

import pytest

from lamplighter.backend.schema import (
    DataNode,
    Graph,
    GraphEdge,
    GraphNode,
    ModelDef,
    ModelLink,
    Project,
)

GRAPH_TS = Path(__file__).resolve().parent.parent / "frontend" / "src" / "types" / "graph.ts"

# Each pydantic model and the TS interface that mirrors its wire shape.
PAIRS = {
    "DomainNode": GraphNode,
    "DomainEdge": GraphEdge,
    "DomainGraph": Graph,
    "DomainModel": ModelDef,
    "DomainLink": ModelLink,
    "DomainDataNode": DataNode,
    "DomainProject": Project,
}

_FIELD = re.compile(r"^\s*(\w+)\??:")


def _ts_interface_fields(source: str, name: str) -> set[str]:
    """The top-level field names of `export interface <name> { ... }`. Nested
    inline object fields (e.g. a `graph: { nodes; edges }`) are skipped by tracking
    brace depth, so only the interface's own keys are returned."""
    m = re.search(rf"export interface {name} \{{", source)
    if m is None:
        raise AssertionError(f"{name} not found in {GRAPH_TS.name}")
    depth = 1  # we're just inside the interface's opening brace
    fields: set[str] = set()
    for line in source[m.end():].splitlines():
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return fields
        if depth == 1:  # a key at the interface's own level, not a nested object
            hit = _FIELD.match(line)
            if hit:
                fields.add(hit.group(1))
    raise AssertionError(f"unterminated interface {name} in {GRAPH_TS.name}")


@pytest.mark.parametrize("ts_name,model", PAIRS.items(), ids=list(PAIRS))
def test_ts_mirror_matches_pydantic(ts_name, model):
    source = GRAPH_TS.read_text()
    ts_fields = _ts_interface_fields(source, ts_name)
    py_fields = set(model.model_fields)
    assert ts_fields == py_fields, (
        f"{ts_name} (graph.ts) drifted from {model.__name__} (schema.py).\n"
        f"  only in TS:      {sorted(ts_fields - py_fields)}\n"
        f"  only in pydantic:{sorted(py_fields - ts_fields)}\n"
        f"Update the mirror so the wire contract stays a straight JSON pass-through."
    )
