"""Design autosave: every mutation writes the design through to a per-project
file (atomically), and session start seeds an empty backend from it — so a
kernel restart with no tab open no longer loses the canvas. Disabled unless a
session configures it, so nothing here touches the filesystem by accident.

The on-disk format wraps the whole Project (``{"version": 2, "project": ...}``);
anything else (including a pre-project bare-graph file) warns and starts blank."""
import json

import pytest

from backend import persist, state
from backend.schema import project_from_graph
from tests.helpers import edge, graph, node


def _mlp():
    return graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )


def _sole_graph(saved):
    """The single model's nodes/edges out of a loaded/saved Project."""
    return saved.models[0].graph


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    """Route the autosave to a temp file and restore the module globals, so
    these tests can't leak state (or files) into the rest of the suite."""
    prior = state.get_project()
    yield
    persist.configure(None)
    state._current = prior


def test_set_project_writes_through_atomically(tmp_path):
    path = tmp_path / "graph.json"
    persist.configure(path)
    g = _mlp()
    state.set_project(project_from_graph(g))

    on_disk = json.loads(path.read_text())
    assert on_disk["version"] == 2
    assert on_disk["project"] == project_from_graph(g).model_dump()
    assert not path.with_suffix(".tmp").exists()  # temp file renamed away

    # Every subsequent mutation overwrites — the file tracks the latest design.
    g2 = _mlp()
    g2.nodes[1].params["out_features"] = 7
    state.set_project(project_from_graph(g2))
    saved = json.loads(path.read_text())
    assert saved["project"]["models"][0]["graph"]["nodes"][1]["params"]["out_features"] == 7


def test_disabled_means_no_writes(tmp_path):
    persist.configure(None)
    state.set_project(project_from_graph(_mlp()))
    assert list(tmp_path.iterdir()) == []


def test_load_round_trips_the_design(tmp_path):
    persist.configure(tmp_path / "graph.json")
    g = _mlp()
    state.set_project(project_from_graph(g))
    assert persist.load().model_dump() == project_from_graph(g).model_dump()


def test_v1_bare_graph_file_warns_and_starts_blank(tmp_path):
    """A pre-project bare-graph file is no longer upgraded (the v1 shim was
    dropped, no-compat) — it takes the corrupt-file path: warn, load None."""
    path = tmp_path / "graph.json"
    persist.configure(path)
    g = _mlp()
    path.write_text(json.dumps(g.model_dump()))  # v1 shape: top-level nodes/edges

    with pytest.warns(UserWarning, match="ignoring the saved project"):
        assert persist.load() is None


def test_missing_corrupt_and_incompatible_files_load_as_none(tmp_path):
    path = tmp_path / "graph.json"
    persist.configure(path)
    assert persist.load() is None  # missing: silent (a fresh project)

    path.write_text("{not json")
    with pytest.warns(UserWarning, match="ignoring the saved project"):
        assert persist.load() is None

    path.write_text(json.dumps({"nodes": "nope"}))  # valid JSON, not a Graph
    with pytest.warns(UserWarning, match="ignoring the saved project"):
        assert persist.load() is None


def test_enable_seeds_an_empty_backend(tmp_path):
    # A previous session saved a design; the kernel restarted (empty state).
    path = tmp_path / "graph.json"
    persist.configure(path)
    g = _mlp()
    persist.save(project_from_graph(g))

    state._current = None
    persist.enable(path)
    assert state.get_project() is not None
    assert state.get_graph().model_dump() == g.model_dump()


def test_enable_never_overwrites_a_live_design(tmp_path):
    # A still-open tab re-seeded the backend before start() ran — the live
    # design wins over the (at best equally fresh) file.
    path = tmp_path / "graph.json"
    persist.configure(path)
    persist.save(project_from_graph(_mlp()))

    live = _mlp()
    live.nodes[1].params["out_features"] = 99
    state.set_project(project_from_graph(live))
    persist.enable(path)
    assert state.get_graph().nodes[1].params["out_features"] == 99


def test_dragend_positions_reach_the_autosave(tmp_path):
    """The moves handler patches positions in place — the write-through must
    still see them, or a restored design would have a stale layout."""
    from backend.app import app
    from fastapi.testclient import TestClient

    path = tmp_path / "graph.json"
    persist.configure(path)
    g = _mlp()

    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        if state.get_project() is not None:
            ws.receive_json()  # the on-connect sync of whatever was cached
        ws.send_json({"type": "validate", "project": project_from_graph(g).model_dump()})
        ws.receive_json()  # shapes reply — the graph is cached and saved
        ws.send_json({"type": "moves", "nodes": [{"id": "l", "position": {"x": 640.0, "y": 5.0}}]})
        # Messages on one socket are handled in order: once code_preview (which
        # never writes the graph) answers, the move before it has been applied
        # and persisted.
        ws.send_json({"type": "code_preview", "enabled": True})
        ws.receive_json()

    saved = json.loads(path.read_text())
    nodes = saved["project"]["models"][0]["graph"]["nodes"]
    moved = next(n for n in nodes if n["id"] == "l")
    assert moved["position"] == {"x": 640.0, "y": 5.0}
