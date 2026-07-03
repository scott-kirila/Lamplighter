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


# --- warm-start resume ----------------------------------------------------------

def _resume(mgr, name, **kwargs):
    """Resume with a deterministic drawn seed (the runner draws from Python's
    random module) and wait for the run to finish."""
    import random

    random.seed(0)
    events: list[dict] = []
    err = mgr.resume(name, checkpoints.load(name), emit=events.append, **kwargs)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    return events


def test_resume_continues_epoch_numbering_and_merges_history():
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)
    stored_history = {k: list(v) for k, v in mgr.history.items()}

    mgr2 = RunManager()
    events = _resume(mgr2, "half", namespace=ns)
    assert mgr2.state == "done"

    # Numbering continues: epochs 13..24 of a planned 24.
    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert [e["epoch"] for e in epochs] == list(range(13, 25))
    assert all(e["epochs"] == 24 for e in epochs)
    assert mgr2.status()["epochs"] == 24

    # One continuous history: the stored 12 epochs, then 12 new ones.
    assert len(mgr2.history["train_loss"]) == 24
    assert mgr2.history["train_loss"][:12] == stored_history["train_loss"]

    # A new seed was drawn and recorded; provenance points at the checkpoint.
    assert mgr2.seed != 3 and mgr2.snapshot["seed"] == mgr2.seed
    assert mgr2.snapshot["resumed_from"] == "half"
    assert mgr2.snapshot["resumed_at_epoch"] == 12
    # The model source travels verbatim from the checkpoint.
    assert mgr2.snapshot["sources"]["model"] == mgr.snapshot["sources"]["model"]


def test_resume_is_a_warm_start_not_a_restart():
    # A tame run (lr 0.1, no re-split surprises on train loss): the first
    # resumed epoch picks up near where training left off, well below a fresh
    # model's starting loss.
    g = _mlp_graph({"epochs": 6, "seed": 5})
    ns = _ns()
    mgr = _trained(g, ns)
    checkpoints.save("warm", manager=mgr)

    mgr2 = RunManager()
    _resume(mgr2, "warm", namespace=ns)
    assert mgr2.history["train_loss"][6] < mgr.history["train_loss"][0]


def test_resume_carries_the_best_across_the_seam():
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    # An unbeatable stored minimum: the marker must survive the resume intact.
    entry = checkpoints.load("half")
    entry["history"]["val_loss"][0] = 1e-9
    stored_best = entry["best_state_dict"]
    mgr2 = RunManager()
    _resume(mgr2, "half", namespace=ns)
    assert mgr2.best_epoch == mgr.best_epoch
    assert all(torch.equal(mgr2.best_state_dict[k], stored_best[k]) for k in stored_best)


def test_resume_claims_the_best_by_beating_the_stored_minimum():
    # The other half: the resumed run's new seed re-splits train/val, so its
    # val loss easily undercuts the stored minimum — the marker must move into
    # the resumed epoch range (offset numbering included).
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    mgr2 = RunManager()
    _resume(mgr2, "half", namespace=ns)
    assert mgr2.best_epoch is not None and mgr2.best_epoch > 12


def test_resume_epochs_override_regenerates_the_trainer():
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    mgr2 = RunManager()
    events = _resume(mgr2, "half", epochs=3, namespace=ns)
    assert [e["epoch"] for e in events if e["type"] == "run_epoch"] == [13, 14, 15]
    assert mgr2.epochs == 15 and len(mgr2.history["train_loss"]) == 15
    assert "epochs=3" in mgr2.snapshot["sources"]["trainer"]


def test_resume_uses_the_checkpoints_own_graph_not_the_last_run():
    from tests.helpers import edge, graph, node

    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("arch-a", manager=mgr)
    stored_source = mgr.snapshot["sources"]["model"]

    # A different architecture runs afterwards — the "canvas" moved on.
    g2 = graph(
        [node("in", "Input", {"shape": "1, 4"}), node("l", "Linear", {"out_features": 2}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    g2.training = {"device": "cpu", "epochs": 1}
    g2.data = {"source": "memory", "x_var": "A", "y_var": "b"}
    torch.manual_seed(0)
    err = mgr.start(g2, namespace={"A": torch.randn(8, 4), "b": torch.randint(0, 2, (8,))},
                    emit=lambda m: None)
    assert err is None and mgr.join(JOIN_TIMEOUT) and mgr.state == "done"

    # Resume still trains checkpoint A's architecture, from its stored source.
    _resume(mgr, "arch-a", namespace=ns)
    assert mgr.state == "done"
    assert mgr.snapshot["sources"]["model"] == stored_source
    with torch.no_grad():
        assert mgr.model.eval()(torch.randn(2, 8)).shape == (2, 3)  # A: 8 → 3


def test_resume_is_refused_mid_run_and_reresolves_data():
    mgr = _trained(_mlp_graph({"epochs": 1}), _ns())
    checkpoints.save("ckpt", manager=mgr)
    entry = checkpoints.load("ckpt")

    # Mid-run: refused with the same message as a second start.
    live = RunManager()
    refusals = []

    def emit(message):
        if message["type"] == "run_epoch" and not refusals:
            refusals.append(live.resume("ckpt", entry))
            live.stop()

    err = live.start(_mlp_graph({"epochs": 50}), _ns(), emit=emit)
    assert err is None and live.join(JOIN_TIMEOUT)
    assert refusals == ["a run is already in progress — stop it first"]

    # Data is re-resolved from the namespace by the stored picks: a dropped
    # name fails pre-flight with the usual message, before anything starts.
    err = live.resume("ckpt", entry, namespace={"y": _ns()["y"]})
    assert err is not None and "not registered" in err


# --- periodic autosave ------------------------------------------------------------

def test_autosave_rolls_a_single_resumable_entry():
    g = _mlp_graph({"epochs": 5, "seed": 1, "autosave_every": 2})
    ns = _ns()
    seen: list = []  # the autosave meta's epoch, observed at each epoch boundary

    def emit(message):
        if message["type"] == "run_epoch":
            autos = [m for m in checkpoints.metas() if m["name"] == "autosave"]
            seen.append(autos[0]["epoch"] if autos else None)

    mgr = RunManager()
    err = mgr.start(g, namespace=ns, emit=emit)
    assert err is None and mgr.join(JOIN_TIMEOUT) and mgr.state == "done"

    # Written every 2 epochs, overwriting: one entry whose epoch advances.
    assert seen == [None, 2, 2, 4, 4]
    assert [m["name"] for m in checkpoints.metas()] == ["autosave"]
    entry = checkpoints.load("autosave")
    assert entry["epoch"] == 4
    assert len(entry["history"]["train_loss"]) == 4

    # The rolling entry is a complete checkpoint: resumable like any other.
    _resume(mgr, "autosave", namespace=ns)
    assert mgr.state == "done"
    assert mgr.snapshot["resumed_from"] == "autosave"
    assert len(mgr.history["train_loss"]) == 4 + 5  # cut at 4, +5 more


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


def test_resume_endpoint_returns_the_preloaded_seam():
    from backend import datastore
    from backend.app import app
    from backend.runner import run_manager
    from fastapi.testclient import TestClient

    g, ns = _overfit_graph()
    datastore.clear()
    datastore.register(**ns)
    try:
        with TestClient(app) as c:
            assert c.post("/api/run/resume", json={"name": "ghost"}).status_code == 404
            _run_via_app(c, g)
            assert c.post("/api/checkpoints", json={"name": "half"}).status_code == 200

            r = c.post("/api/run/resume", json={"name": "half"})
            assert r.status_code == 200
            body = r.json()
            # The starting status carries the stored curve, so the acting tab
            # seeds its charts with the seam already in place.
            assert body["state"] == "running" and body["epochs"] == 24
            assert len(body["history"]["train_loss"]) >= 12

            assert run_manager.join(JOIN_TIMEOUT)
            status = c.get("/api/run/status").json()
            assert status["state"] == "done"
            assert len(status["history"]["train_loss"]) == 24
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
