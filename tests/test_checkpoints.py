"""The session's checkpoint store: named in-kernel snapshots of finished runs,
saved/listed/restored from both the notebook and the app. Store CRUD is tested
against injected RunManagers; the REST + WS layers against the real app (which
uses the singleton run_manager, as production does)."""
import pytest
import torch
import torch.nn as nn

from backend import checkpoints
from backend.runner import RunManager
from tests.test_runner import JOIN_TIMEOUT, _mlp_graph, _ns, _overfit_graph, _start


@pytest.fixture(autouse=True)
def _clean_store():
    checkpoints.clear()
    yield
    checkpoints.clear()


def _trained(graph=None, ns=None):
    """A RunManager that just finished the given run (default: the overfit
    graph, whose best epoch differs from its final one)."""
    if graph is None:
        graph, ns = _overfit_graph()
    mgr, _, err = _start(graph, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done"
    return mgr


# --- store CRUD ---------------------------------------------------------------

def test_save_requires_a_trained_model():
    with pytest.raises(ValueError, match="no trained model"):
        checkpoints.save("a", manager=RunManager())


def test_save_requires_a_name():
    with pytest.raises(ValueError, match="name must not be empty"):
        checkpoints.save("   ", manager=_trained(_mlp_graph({"epochs": 1}), _ns()))


def test_save_lists_the_runs_identity_and_metrics():
    mgr = _trained()
    meta = checkpoints.save("overfit", manager=mgr)
    assert checkpoints.metas() == [meta]
    assert meta["name"] == "overfit"
    assert meta["epoch"] == 12
    assert meta["best_epoch"] == mgr.best_epoch
    assert meta["seed"] == 3
    assert meta["val_loss"] == mgr.history["val_loss"][-1]
    assert "T" in meta["created"]  # ISO timestamp


def test_save_overwrites_the_same_name():
    checkpoints.save("ckpt", manager=_trained(_mlp_graph({"epochs": 2}), _ns()))
    checkpoints.save("ckpt", manager=_trained(_mlp_graph({"epochs": 4}), _ns()))
    metas = checkpoints.metas()
    assert len(metas) == 1
    assert metas[0]["epoch"] == 4  # the newer run replaced the older


def test_stored_weights_are_isolated_from_the_live_model():
    # sess.model stays reachable and mutable (further training, notebook
    # fine-tuning) — the stored entry must be a copy, not references.
    mgr = _trained(_mlp_graph({"epochs": 1}), _ns())
    checkpoints.save("frozen", manager=mgr)
    before = {k: v.clone() for k, v in checkpoints.load("frozen")["state_dict"].items()}
    with torch.no_grad():
        for p in mgr.model.parameters():
            p.add_(1.0)
    after = checkpoints.load("frozen")["state_dict"]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_load_and_delete_reject_unknown_names():
    checkpoints.save("only", manager=_trained(_mlp_graph({"epochs": 1}), _ns()))
    with pytest.raises(ValueError, match="no checkpoint named 'nope'.*only"):
        checkpoints.load("nope")
    with pytest.raises(ValueError, match="no checkpoint named 'nope'"):
        checkpoints.delete("nope")
    checkpoints.delete("only")
    assert checkpoints.metas() == []


# --- restore ------------------------------------------------------------------

def test_restore_repopulates_the_run_artifacts():
    # Save the overfit run, run something ELSE on the same manager, restore —
    # everything (model, history, snapshot, seed, best) is the saved run again.
    mgr = _trained()
    checkpoints.save("keep", manager=mgr)
    saved_status = mgr.status()
    x = torch.randn(4, 8)
    with torch.no_grad():
        saved_out = mgr.model.eval()(x)
        saved_best_out = mgr.best_model()(x)

    mgr2, _, err = _start(_mlp_graph({"epochs": 2, "seed": 99}), _ns())
    assert err is None and mgr2.join(JOIN_TIMEOUT)
    # (fresh manager — mirror production where restore hits whatever ran last)
    assert mgr2.restore(checkpoints.load("keep")) is None

    assert mgr2.status() == saved_status
    assert mgr2.seed == 3 and mgr2.best_epoch == mgr.best_epoch
    assert mgr2.snapshot["sources"]["model"] == mgr.snapshot["sources"]["model"]
    with torch.no_grad():
        assert torch.equal(mgr2.model(x), saved_out)  # restore() leaves it in eval
        assert torch.equal(mgr2.best_model()(x), saved_best_out)


def test_restore_is_refused_mid_run():
    mgr = _trained(_mlp_graph({"epochs": 1}), _ns())
    checkpoints.save("ckpt", manager=mgr)
    entry = checkpoints.load("ckpt")

    live = RunManager()
    refusals = []

    def emit(message):
        if message["type"] == "run_epoch" and not refusals:
            refusals.append(live.restore(entry))
            live.stop()

    err = live.start(_mlp_graph({"epochs": 50}), _ns(), emit=emit)
    assert err is None and live.join(JOIN_TIMEOUT)
    assert refusals == ["a run is in progress — stop it before restoring a checkpoint"]


# --- REST + WS (the app's paths, via the singleton run manager) ----------------

def _run_via_app(client, graph):
    from backend.runner import run_manager

    r = client.post("/api/run/start", json=graph.model_dump())
    assert r.status_code == 200
    assert run_manager.join(JOIN_TIMEOUT)


def test_checkpoint_endpoints_end_to_end(tmp_path):
    """Save → list → download (loadable anywhere, best included) → restore →
    delete, through the REST surface the app uses."""
    import lamplighter
    from backend import datastore
    from backend.app import app
    from fastapi.testclient import TestClient

    g, ns = _overfit_graph()
    datastore.clear()
    datastore.register(**ns)
    try:
        with TestClient(app) as c:
            # An empty/blank name is rejected regardless of run state.
            assert c.post("/api/checkpoints", json={"name": "  "}).status_code == 400

            _run_via_app(c, g)
            r = c.post("/api/checkpoints", json={"name": "best-run"})
            assert r.status_code == 200
            assert r.json()["checkpoint"]["epoch"] == 12

            listing = c.get("/api/checkpoints").json()["checkpoints"]
            assert [m["name"] for m in listing] == ["best-run"]

            # The download round-trips through load_checkpoint — best included.
            r = c.get("/api/checkpoints/best-run/weights")
            assert r.status_code == 200
            assert r.headers["content-disposition"] == 'attachment; filename="best-run.pt"'
            path = tmp_path / "best-run.pt"
            path.write_bytes(r.content)
            rebuilt, snapshot = lamplighter.load_checkpoint(str(path), best=True)
            assert snapshot["seed"] == 3
            assert isinstance(rebuilt, nn.Module)

            # Run something else, then restore — status is the saved run again.
            _run_via_app(c, _mlp_graph({"epochs": 2, "seed": 99}))
            assert c.get("/api/run/status").json()["epochs"] == 2
            r = c.post("/api/checkpoints/best-run/restore")
            assert r.status_code == 200
            assert r.json()["epochs"] == 12 and r.json()["seed"] == 3
            assert c.get("/api/run/status").json() == r.json()

            # Unknown names 404; delete empties the listing.
            assert c.post("/api/checkpoints/nope/restore").status_code == 404
            assert c.get("/api/checkpoints/nope/weights").status_code == 404
            assert c.delete("/api/checkpoints/nope").status_code == 404
            assert c.delete("/api/checkpoints/best-run").status_code == 200
            assert c.get("/api/checkpoints").json()["checkpoints"] == []
    finally:
        datastore.clear()


def test_store_changes_push_the_listing_over_the_websocket():
    from backend import datastore
    from backend.app import app
    from fastapi.testclient import TestClient

    datastore.clear()
    datastore.register(**_ns())
    try:
        with TestClient(app) as c:
            _run_via_app(c, _mlp_graph({"epochs": 1}))
            with c.websocket_connect("/ws") as ws:
                assert c.post("/api/checkpoints", json={"name": "live"}).status_code == 200
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "checkpoints":
                        break
                assert [m["name"] for m in msg["checkpoints"]] == ["live"]
    finally:
        datastore.clear()


def test_session_checkpoint_methods():
    """The notebook side: sess.checkpoint/checkpoints/restore drive the same
    store the app's strip shows."""
    from backend import datastore
    from backend.app import app
    from backend.runner import run_manager
    from fastapi.testclient import TestClient
    from lamplighter import LamplighterError
    from lamplighter.session import Session

    sess = Session("127.0.0.1", 1)  # in-process; no server thread needed
    g, ns = _overfit_graph()
    datastore.clear()
    try:
        sess.data(**ns)
        with pytest.raises(LamplighterError, match="no checkpoint named"):
            sess.restore("nothing")
        with TestClient(app) as c:
            _run_via_app(c, g)
        meta = sess.checkpoint("from-notebook")
        assert meta["best_epoch"] == run_manager.best_epoch
        assert [m["name"] for m in sess.checkpoints()] == ["from-notebook"]

        status = sess.restore("from-notebook")
        assert status["state"] == "done" and status["epochs"] == 12
        assert isinstance(sess.model, nn.Module)
        assert isinstance(sess.best_model, nn.Module)
    finally:
        datastore.clear()
