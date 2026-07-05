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
from backend.inference import infer_shapes, link_issues, primary_shapes
from backend.schema import Graph, ModelDef, ModelLink, Project
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


def test_multi_model_code_panel_uses_per_model_class_names():
    from backend.ws import _validate_project

    _, code, _ = _validate_project(_two_model_project(), want_code=True)
    assert "class Generator(nn.Module):" in code["g"]
    assert "class Discriminator(nn.Module):" in code["d"]


def test_single_model_code_panel_stays_generatedmodel():
    from backend.schema import project_from_graph
    from backend.ws import _validate_project

    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    _, code, _ = _validate_project(project_from_graph(g), want_code=True)
    (only,) = code.values()
    assert "class GeneratedModel(nn.Module):" in only


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


def _shapes_for(project):
    """Per-model primary shapes, as _validate_project feeds link_issues."""
    out = {}
    for m in project.models:
        shapes, _ = infer_shapes(m.graph)
        out[m.id] = primary_shapes(m.graph, shapes)
    return out


def test_link_ok_when_output_matches_input():
    # Generator: 100 -> 784 ; Discriminator input: 784. Output feeds input.
    gen = graph(
        [node("in", "Input", {"shape": "1, 100"}), node("l", "Linear", {"out_features": 784}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    disc = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("l", "Linear", {"out_features": 1}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    project = Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        links=[ModelLink(id="L", source_model="g", target_model="d")],
    )
    (result,) = link_issues(project, _shapes_for(project))
    assert result["ok"] is True
    assert result["message"] == "Generator → Discriminator: N × 784"


def test_link_flags_a_shape_mismatch():
    gen = graph(
        [node("in", "Input", {"shape": "1, 100"}), node("l", "Linear", {"out_features": 256}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    disc = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("l", "Linear", {"out_features": 1}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    project = Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        links=[ModelLink(id="L", source_model="g", target_model="d")],
    )
    (result,) = link_issues(project, _shapes_for(project))
    assert result["ok"] is False
    assert "Generator output N × 256 ≠ Discriminator input N × 784" == result["message"]


def _disc(in_shape="1, 8"):
    from backend.schema import Graph, ModelDef

    g = graph(
        [node("in", "Input", {"shape": in_shape}), node("l", "Linear", {"out_features": 1}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return ModelDef(id="d", name="Discriminator", graph=Graph(nodes=g.nodes, edges=g.edges))


def test_data_link_round_trips_and_checks_the_dataset_against_the_input():
    import torch
    from backend.inference import data_node_output_shape
    from backend.schema import DataNode

    dn = DataNode(id="x", kind="dataset", name="MNIST", config={"source": "memory", "x_var": "X"})
    project = Project(
        models=[_disc("1, 8")],
        data_nodes=[dn],
        links=[ModelLink(id="L", source_data="x", target_model="d")],
    )
    # The data link round-trips (source_data set, source_model None).
    assert Project.model_validate(project.model_dump()).links[0].source_data == "x"

    ns = {"X": torch.randn(20, 8)}
    data_shapes = {"x": data_node_output_shape(dn, ns)}  # X (20, 8) → [1, 8]
    assert data_shapes["x"] == [1, 8]
    (res,) = link_issues(project, _shapes_for(project), data_shapes)
    assert res["ok"] is True and "MNIST → Discriminator: N × 8" == res["message"]


def test_data_link_flags_a_dataset_shape_mismatch():
    import torch
    from backend.inference import data_node_output_shape
    from backend.schema import DataNode

    dn = DataNode(id="x", kind="dataset", name="MNIST", config={"source": "memory", "x_var": "X"})
    project = Project(
        models=[_disc("1, 784")],  # expects 784, but X is 8-dim
        data_nodes=[dn],
        links=[ModelLink(id="L", source_data="x", target_model="d")],
    )
    ns = {"X": torch.randn(20, 8)}
    (res,) = link_issues(project, _shapes_for(project), {"x": data_node_output_shape(dn, ns)})
    assert res["ok"] is False and "≠" in res["message"]


def test_noise_node_output_shape_from_dims():
    from backend.inference import data_node_output_shape
    from backend.schema import DataNode

    assert data_node_output_shape(DataNode(id="n", kind="noise", config={"dims": "100"}), {}) == [1, 100]
    assert data_node_output_shape(DataNode(id="n", kind="noise", config={"dims": "100, 1, 1"}), {}) == [1, 100, 1, 1]


def test_unresolved_data_link_shows_a_neutral_wire():
    from backend.schema import DataNode

    # A dataset with nothing picked → shape unknown → the wire shows, no verdict.
    dn = DataNode(id="x", kind="dataset", name="Data", config={"source": "memory"})
    project = Project(
        models=[_disc("1, 8")],
        data_nodes=[dn],
        links=[ModelLink(id="L", source_data="x", target_model="d")],
    )
    (res,) = link_issues(project, _shapes_for(project))  # no data_shapes
    assert res["ok"] is True and res["message"] == "Data → Discriminator"


def test_link_check_rides_the_ws_payload():
    gen = graph(
        [node("in", "Input", {"shape": "1, 100"}), node("l", "Linear", {"out_features": 784}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    disc = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("l", "Linear", {"out_features": 1}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    project = Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        links=[ModelLink(id="L", source_model="g", target_model="d")],
    )
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "validate", "project": project.model_dump()})
        msg = ws.receive_json()
    assert msg["links"] == [{"id": "L", "ok": True, "message": "Generator → Discriminator: N × 784"}]


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
