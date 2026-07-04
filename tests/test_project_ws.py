"""The project-shaped WebSocket protocol: a whole project (multiple models)
validates, syncs, and moves without collapsing to a single graph.

Guards the multi-model contract — a `validate` carrying two models must keep
both (an earlier single-graph `validate` would have clobbered all but one), with
per-model shape results and model-scoped drag persistence.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend import persist, state
from backend.app import app
from backend.schema import Graph, ModelDef, Project
from tests.helpers import edge, graph, node


def _model(mid, name, out_features):
    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": out_features}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return ModelDef(id=mid, name=name, graph=Graph(nodes=g.nodes, edges=g.edges))


def _two_model_project():
    return Project(
        models=[_model("g", "Generator", 4), _model("d", "Discriminator", 1)],
        training={"lr": 0.1},
    )


@pytest.fixture(autouse=True)
def _isolated():
    prior = state.get_project()
    state._current = None
    persist.configure(None)
    yield
    persist.configure(None)
    state._current = prior


def test_validate_keeps_every_model_with_per_model_shapes():
    project = _two_model_project()
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "validate", "project": project.model_dump()})
        msg = ws.receive_json()

    assert msg["type"] == "shapes"
    # Both models are inferred, each keyed by its own id.
    assert set(msg["models"]) == {"g", "d"}
    assert msg["models"]["g"]["shapes"]["l"] == [1, 4]
    assert msg["models"]["d"]["shapes"]["l"] == [1, 1]
    # The backend stored the whole project, not a single collapsed graph.
    stored = state.get_project()
    assert [m.id for m in stored.models] == ["g", "d"]


def test_on_connect_sync_carries_the_project():
    state.set_project(_two_model_project())
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "sync"
    assert [m["id"] for m in msg["project"]["models"]] == ["g", "d"]
    assert set(msg["models"]) == {"g", "d"}


def test_model_scoped_moves_patch_only_that_model():
    state.set_project(_two_model_project())
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()  # on-connect sync
        ws.send_json({
            "type": "moves",
            "model_id": "d",
            "nodes": [{"id": "l", "position": {"x": 999.0, "y": 7.0}}],
        })
        # A follow-up validate serializes after the move is applied.
        ws.send_json({"type": "code_preview", "enabled": True})
        ws.receive_json()

    stored = state.get_project()
    d = next(m for m in stored.models if m.id == "d")
    g = next(m for m in stored.models if m.id == "g")
    assert next(n for n in d.graph.nodes if n.id == "l").position.x == 999.0
    # The other model's identically-named node is untouched.
    assert next(n for n in g.graph.nodes if n.id == "l").position.x == 0.0


def test_system_moves_patch_model_positions_and_persist(tmp_path):
    path = tmp_path / "project.json"
    persist.configure(path)
    state.set_project(_two_model_project())
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()  # on-connect sync
        ws.send_json({"type": "system_moves", "nodes": [{"id": "d", "position": {"x": 50.0, "y": 80.0}}]})
        ws.send_json({"type": "code_preview", "enabled": True})
        ws.receive_json()

    saved = json.loads(path.read_text())
    d = next(m for m in saved["project"]["models"] if m["id"] == "d")
    assert d["sys_position"] == {"x": 50.0, "y": 80.0}
