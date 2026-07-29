"""The session's checkpoint store: named in-kernel snapshots of finished runs,
saved/listed/restored from both the notebook and the app. Store CRUD is tested
against injected RunManagers; the REST + WS layers against the real app (which
uses the singleton run_manager, as production does)."""
import warnings

import pytest
import torch
import torch.nn as nn

from lamplighter.backend import checkpoints
from lamplighter.backend.runner import RunManager
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
    assert meta["epochs"] == 12  # the plan — epoch < epochs marks an interrupted run
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
    before = {k: v.clone() for k, v in checkpoints.load("frozen")["state_dicts"]["model"].items()}
    with torch.no_grad():
        for p in mgr.model.parameters():
            p.add_(1.0)
    after = checkpoints.load("frozen")["state_dicts"]["model"]
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

    # The saved artifacts round-trip, per-layer health curve included: it's now a
    # checkpoint artifact, so a restored run shows the same health it had (and can
    # resume without the health panel resetting).
    assert mgr2.status()["health_history"] == saved_status["health_history"]
    assert mgr2.status()["health_history"]  # non-empty — the run produced health
    assert mgr2.status() == saved_status
    assert mgr2.seed == 3 and mgr2.best_epoch == mgr.best_epoch
    assert mgr2.snapshot["sources"]["models"]["model"] == mgr.snapshot["sources"]["models"]["model"]
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
    events = _resume(mgr2, "half", epochs=24, namespace=ns)  # extend: new total target
    assert mgr2.state == "done"

    # Numbering continues: epochs 13..24 of a planned 24.
    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert [e["epoch"] for e in epochs] == list(range(13, 25))
    assert all(e["epochs"] == 24 for e in epochs)
    assert mgr2.status()["epochs"] == 24

    # One continuous history: the stored 12 epochs, then 12 new ones.
    assert len(mgr2.history["train_loss"]) == 24
    assert mgr2.history["train_loss"][:12] == stored_history["train_loss"]

    # The per-layer health curve continues across the seam too (the checkpoint's
    # 12 snapshots, then 12 new) — the health panel doesn't reset on resume.
    resumed_health = mgr2.status()["health_history"]
    assert len(resumed_health) == 24
    assert resumed_health[:12] == mgr.status()["health_history"]

    # A new seed was drawn and recorded; provenance points at the checkpoint.
    assert mgr2.seed != 3 and mgr2.snapshot["seed"] == mgr2.seed
    assert mgr2.snapshot["resumed_from"] == "half"
    assert mgr2.snapshot["resumed_at_epoch"] == 12
    # The model source travels verbatim from the checkpoint.
    assert mgr2.snapshot["sources"]["models"]["model"] == mgr.snapshot["sources"]["models"]["model"]


def test_resume_is_a_warm_start_not_a_restart():
    # A tame run (lr 0.1, no re-split surprises on train loss): the first
    # resumed epoch picks up near where training left off, well below a fresh
    # model's starting loss.
    g = _mlp_graph({"epochs": 6, "seed": 5})
    ns = _ns()
    mgr = _trained(g, ns)
    checkpoints.save("warm", manager=mgr)

    mgr2 = RunManager()
    _resume(mgr2, "warm", epochs=12, namespace=ns)
    assert mgr2.history["train_loss"][6] < mgr.history["train_loss"][0]


def test_resume_finishes_an_interrupted_plan():
    # The recovery workflow: a run stopped at epoch 1 of 6 resumes with NO
    # arguments and finishes exactly where it was headed — epoch 6, not 7.
    mgr = RunManager()

    def emit(message):
        if message["type"] == "run_epoch":
            mgr.stop()

    err = mgr.start(_mlp_graph({"epochs": 6, "seed": 5}), _ns(), emit=emit)
    assert err is None and mgr.join(JOIN_TIMEOUT) and mgr.state == "stopped"
    meta = checkpoints.save("cut", manager=mgr)
    assert meta["epoch"] == 1 and meta["epochs"] == 6  # visibly interrupted

    events = _resume(mgr, "cut", namespace=_ns())
    assert mgr.state == "done"
    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert [e["epoch"] for e in epochs] == [2, 3, 4, 5, 6]
    assert all(e["epochs"] == 6 for e in epochs)
    assert len(mgr.history["train_loss"]) == 6


def test_resume_refuses_a_completed_plan_without_a_target():
    ns = _ns()
    mgr = _trained(_mlp_graph({"epochs": 2}), ns)
    checkpoints.save("done", manager=mgr)
    entry = checkpoints.load("done")

    fresh = RunManager()
    err = fresh.resume("done", entry, namespace=ns)
    assert err == (
        "'done' already completed its 2-epoch plan — pass a higher target to "
        "train further, e.g. epochs=4"
    )
    # An explicit target must actually be past what's trained.
    err = fresh.resume("done", entry, epochs=2, namespace=ns)
    assert err is not None and "not past the 2 epochs already trained" in err


def test_resume_carries_the_best_across_the_seam():
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    # An unbeatable stored minimum: the marker must survive the resume intact.
    entry = checkpoints.load("half")
    entry["history"]["val_loss"][0] = 1e-9
    stored_best = entry["best_state_dict"]
    mgr2 = RunManager()
    _resume(mgr2, "half", epochs=24, namespace=ns)
    assert mgr2.best_epoch == mgr.best_epoch
    assert all(torch.equal(mgr2.best_state_dict[k], stored_best[k]) for k in stored_best)


def test_resume_claims_the_best_by_beating_the_stored_minimum():
    # The other half of the marker machinery: when a resumed epoch undercuts the
    # stored minimum, the best marker moves into the resumed range (offset
    # numbering included). The resume validates on the *same* (stable) split as
    # the stored run, so we make the stored minimum trivially beatable rather than
    # relying on a lucky re-split — a resumed epoch then legitimately claims it.
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    entry = checkpoints.load("half")
    entry["history"]["val_loss"] = [1e9] * len(entry["history"]["val_loss"])
    mgr2 = RunManager()
    _resume(mgr2, "half", epochs=24, namespace=ns)
    assert mgr2.best_epoch is not None and mgr2.best_epoch > 12


def test_resume_bakes_the_whole_plan_and_the_pickup_point_into_the_trainer():
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("half", manager=mgr)

    mgr2 = RunManager()
    events = _resume(mgr2, "half", epochs=15, namespace=ns)  # 12 done, target 15
    assert [e["epoch"] for e in events if e["type"] == "run_epoch"] == [13, 14, 15]
    assert mgr2.epochs == 15 and len(mgr2.history["train_loss"]) == 15
    # The trainer used to bake in the REMAINING count ("epochs=3"), which made
    # any LR schedule restart its shape for the second segment. It now carries
    # the whole plan plus where this segment picks up, so a schedule spans the
    # run it was configured for — and the source reads as what happened.
    trainer = mgr2.snapshot["sources"]["trainer"]
    assert "epochs=15" in trainer and "start_epoch=12" in trainer
    assert mgr2.snapshot["training"]["epochs"] == 15


def test_resume_uses_the_checkpoints_own_graph_not_the_last_run():
    from tests.helpers import edge, graph, node, single_model_project

    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("arch-a", manager=mgr)
    stored_source = mgr.snapshot["sources"]["models"]["model"]

    # A different architecture runs afterwards — the "canvas" moved on.
    g2 = graph(
        [node("in", "Input", {"shape": "1, 4"}), node("l", "Linear", {"out_features": 2}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    project2 = single_model_project(
        g2, training={"device": "cpu", "epochs": 1},
        data={"source": "memory", "x_var": "A", "y_var": "b"},
    )
    torch.manual_seed(0)
    err = mgr.start(project2, namespace={"A": torch.randn(8, 4), "b": torch.randint(0, 2, (8,))},
                    emit=lambda m: None)
    assert err is None and mgr.join(JOIN_TIMEOUT) and mgr.state == "done"

    # Resume still trains checkpoint A's architecture, from its stored source.
    _resume(mgr, "arch-a", epochs=24, namespace=ns)
    assert mgr.state == "done"
    assert mgr.snapshot["sources"]["models"]["model"] == stored_source
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
            refusals.append(live.resume("ckpt", entry, epochs=3))
            live.stop()

    err = live.start(_mlp_graph({"epochs": 50}), _ns(), emit=emit)
    assert err is None and live.join(JOIN_TIMEOUT)
    assert refusals == ["a run is already in progress — stop it first"]

    # Data is re-resolved from the namespace by the stored picks: a dropped
    # name fails pre-flight with the usual message, before anything starts.
    err = live.resume("ckpt", entry, epochs=3, namespace={"y": _ns()["y"]})
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

    # The rolling entry is a complete checkpoint, and no-argument resume
    # finishes its interrupted plan: cut at 4 of 5, one epoch remains.
    _resume(mgr, "autosave", namespace=ns)
    assert mgr.state == "done"
    assert mgr.snapshot["resumed_from"] == "autosave"
    assert len(mgr.history["train_loss"]) == 5
    assert mgr.epochs == 5


# --- REST + WS (the app's paths, via the singleton run manager) ----------------

def _run_via_app(client, graph):
    from lamplighter.backend.runner import run_manager

    r = client.post("/api/run/start", json=graph.model_dump())
    assert r.status_code == 200
    assert run_manager.join(JOIN_TIMEOUT)


def test_checkpoint_endpoints_end_to_end(tmp_path):
    """Save → list → download (loadable anywhere, best included) → restore →
    delete, through the REST surface the app uses."""
    import lamplighter
    from lamplighter.backend import datastore
    from lamplighter.backend.app import app
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

            # The run auto-recorded itself (weightless run-1); the save added
            # the weighted entry.
            listing = c.get("/api/checkpoints").json()["checkpoints"]
            assert [m["name"] for m in listing] == ["run-1", "best-run"]
            assert [m["has_weights"] for m in listing] == [False, True]

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
            # The auto records of the two runs remain — deleting a named save
            # never touches the run history.
            names = [m["name"] for m in c.get("/api/checkpoints").json()["checkpoints"]]
            assert names == ["run-1", "run-2"]
    finally:
        datastore.clear()


def test_resume_endpoint_returns_the_preloaded_seam():
    from lamplighter.backend import datastore
    from lamplighter.backend.app import app
    from lamplighter.backend.runner import run_manager
    from fastapi.testclient import TestClient

    g, ns = _overfit_graph()
    datastore.clear()
    datastore.register(**ns)
    try:
        with TestClient(app) as c:
            assert c.post("/api/run/resume", json={"name": "ghost"}).status_code == 404
            _run_via_app(c, g)
            assert c.post("/api/checkpoints", json={"name": "half"}).status_code == 200

            # The plan is complete — resuming without a target is a 400 that
            # spells out the fix.
            r = c.post("/api/run/resume", json={"name": "half"})
            assert r.status_code == 400
            assert "already completed its 12-epoch plan" in r.json()["detail"]

            r = c.post("/api/run/resume", json={"name": "half", "epochs": 24})
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
    from lamplighter.backend import datastore
    from lamplighter.backend.app import app
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
                assert [m["name"] for m in msg["checkpoints"]] == ["run-1", "live"]
    finally:
        datastore.clear()


def test_session_checkpoint_methods():
    """The notebook side: sess.checkpoint/checkpoints/restore drive the same
    store the app's strip shows."""
    from lamplighter.backend import datastore
    from lamplighter.backend.app import app
    from lamplighter.backend.runner import run_manager
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
        assert [m["name"] for m in sess.checkpoints()] == ["run-1", "from-notebook"]

        status = sess.restore("from-notebook")
        assert status["state"] == "done" and status["epochs"] == 12
        assert isinstance(sess.model, nn.Module)
        assert isinstance(sess.best_model, nn.Module)
    finally:
        datastore.clear()


def test_checkpoint_history_endpoint_serves_curves_and_config():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    # The endpoint reads the module-global store; put a real run's entry in it.
    g, ns = _overfit_graph()
    checkpoints.save("cmp", manager=_trained(g, ns))

    with TestClient(app) as c:
        body = c.get("/api/checkpoints/cmp/history").json()
        assert body["name"] == "cmp"
        # The full curves (what the overlay draws) + the config (the diff table).
        assert len(body["history"]["train_loss"]) == 12
        assert len(body["history"]["val_loss"]) == 12
        assert body["training"]["epochs"] == 12 and body["training"]["seed"] == 3
        # The recorded data config rides along — the diff table shows data-axis
        # differences (batch size, augmentations), not just training params.
        assert body["data"]["source"] == "memory"
        assert "batch_size" in body["data"]

        assert c.get("/api/checkpoints/nope/history").status_code == 404


# --- persistence: saved runs survive a kernel restart ---------------------------

def _tiny_checkpoint(epoch=1):
    return {
        "state_dicts": {"model": {"w": torch.ones(2, 2)}},
        "best_state_dict": None,
        "best_epoch": None,
        "epoch": epoch,
        "history": {"train_loss": [1.0] * epoch},
        "snapshot": {"training": {"epochs": epoch}, "seed": 7},
    }


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "checkpoints"
    checkpoints.configure(d)
    yield d
    checkpoints.configure(None)


def _simulate_kernel_restart(d):
    checkpoints._store.clear()  # a fresh kernel: nothing in memory
    checkpoints.enable(d)


def test_checkpoints_survive_a_kernel_restart(ckpt_dir):
    # The real loop: train, save, "restart", and the run restores from disk.
    mgr = _trained()
    checkpoints.save("keep", manager=mgr)
    x = torch.randn(4, 8)
    with torch.no_grad():
        expected = mgr.model.eval()(x)

    _simulate_kernel_restart(ckpt_dir)
    (meta,) = checkpoints.metas()
    assert meta["name"] == "keep" and meta["epoch"] == 12  # listed from the sidecar

    fresh = RunManager()
    assert fresh.restore(checkpoints.load("keep")) is None
    with torch.no_grad():
        assert torch.equal(fresh.model(x), expected)


def test_hydration_is_lazy(ckpt_dir):
    checkpoints.save_entry("a", _tiny_checkpoint())
    _simulate_kernel_restart(ckpt_dir)
    assert checkpoints._store["a"]["checkpoint"] is None  # listed, not loaded
    checkpoints.metas()  # listing never touches the weights
    assert checkpoints._store["a"]["checkpoint"] is None
    assert checkpoints.load("a")["epoch"] == 1  # first use materializes
    assert checkpoints._store["a"]["checkpoint"] is not None


def test_delete_removes_the_files(ckpt_dir):
    checkpoints.save_entry("gone", _tiny_checkpoint())
    assert len(list(ckpt_dir.iterdir())) == 2  # .pt + .json sidecar
    checkpoints.delete("gone")
    assert list(ckpt_dir.iterdir()) == []


def test_disabled_means_no_checkpoint_files(tmp_path):
    checkpoints.configure(None)
    checkpoints.save_entry("mem-only", _tiny_checkpoint())
    assert list(tmp_path.iterdir()) == []


def test_live_entry_wins_over_its_file(ckpt_dir):
    checkpoints.save_entry("x", _tiny_checkpoint(epoch=1))
    checkpoints._store["x"]["checkpoint"]["epoch"] = 99  # the live (fresher) state
    checkpoints.enable(ckpt_dir)  # re-hydration must not clobber it
    assert checkpoints.load("x")["epoch"] == 99


def test_awkward_names_round_trip(ckpt_dir):
    name = "run 1/α β"
    checkpoints.save_entry(name, _tiny_checkpoint())
    _simulate_kernel_restart(ckpt_dir)
    assert checkpoints.metas()[0]["name"] == name
    assert checkpoints.load(name)["epoch"] == 1


def test_corrupt_sidecar_warns_and_skips(ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "junk.json").write_text("{not json")
    with pytest.warns(UserWarning, match="ignoring the saved checkpoint"):
        checkpoints.enable(ckpt_dir)
    assert checkpoints.metas() == []


def test_hydration_preserves_chronological_order(ckpt_dir):
    # 12 runs: the sidecar FILES sort lexicographically (run-10 before run-2),
    # but the listing's insertion order is its chronology — the frontend shows
    # newest-first by reversing it and "latest run" reads the tail — so
    # hydration must sort by each row's created stamp, not by filename.
    for i in range(1, 13):
        ckpt = _tiny_checkpoint()
        ckpt["snapshot"]["seed"] = i
        checkpoints._store[f"run-{i}"] = {
            "checkpoint": ckpt, "created": f"2026-07-19T00:00:{i:02d}", "auto": True,
        }
        checkpoints._write_through(f"run-{i}", checkpoints._store[f"run-{i}"])
    before = [m["name"] for m in checkpoints.metas()]

    _simulate_kernel_restart(ckpt_dir)
    assert [m["name"] for m in checkpoints.metas()] == before  # run-1 … run-12


def test_run_counter_file_is_not_mistaken_for_a_sidecar(ckpt_dir):
    # next_run_name persists its counter as run-counter.json in the same dir;
    # hydration must skip it silently, not warn "ignoring the saved checkpoint".
    checkpoints.save_entry("run-1", _tiny_checkpoint())
    checkpoints.next_run_name()  # writes run-counter.json
    checkpoints._store.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        checkpoints.enable(ckpt_dir)
    assert [m["name"] for m in checkpoints.metas()] == ["run-1"]


# --- what a resumed segment inherits, and what it must not ----------------------

def test_early_stopped_run_resumes_for_real():
    """The stall was measured as (epoch - best_epoch), and best_epoch is
    RESTORED from the previous segment — so a run that stopped at 5 with its
    best at 3 resumed at 21, computed a stall of 18, stopped after one epoch and
    reported "done". Resuming is an explicit request to keep going."""
    g, ns = _overfit_graph()
    # A divergent lr pins the best early; patience 2 ends the first segment well
    # short of its 30-epoch plan.
    mgr = RunManager()
    err = mgr.start(
        _mlp_graph({"epochs": 30, "lr": 25.0, "seed": 0, "early_stop_patience": 2},
                   data={"val_split": 0.25}),
        _ns(n=32), emit=lambda m: None,
    )
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done" and len(mgr.history["train_loss"]) < 30
    first = len(mgr.history["train_loss"])
    checkpoints.save("stalled", manager=mgr)

    mgr2 = RunManager()
    _resume(mgr2, "stalled", epochs=first + 10, namespace=_ns(n=32))
    trained = len(mgr2.history["train_loss"]) - first
    assert trained > 1, f"resumed segment trained only {trained} epoch(s)"


def test_a_resumed_segment_gets_a_fresh_patience_budget():
    """Resume on the SAME manager that just early-stopped — the production
    singleton path — so the stall counter is genuinely inherited. A fresh
    RunManager cannot detect a missing reset (``__init__`` zeroes the counter),
    which is exactly how the old version of this test could never fail."""
    mgr = RunManager()
    err = mgr.start(
        _mlp_graph({"epochs": 30, "lr": 25.0, "seed": 0, "early_stop_patience": 2},
                   data={"val_split": 0.25}),
        _ns(n=32), emit=lambda m: None,
    )
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done" and len(mgr.history["train_loss"]) < 30
    assert mgr._epochs_since_best >= 2  # the stall that ended the segment, still live
    first = len(mgr.history["train_loss"])
    checkpoints.save("stalled-inplace", manager=mgr)

    _resume(mgr, "stalled-inplace", epochs=first + 10, namespace=_ns(n=32))
    trained = len(mgr.history["train_loss"]) - first
    assert trained > 1, f"the inherited stall ended the resumed segment after {trained} epoch(s)"


def test_resume_keeps_the_recipes_validation_contract():
    """generate_dataloader was called without has_val on the resume path, so it
    defaulted to True: a recipe that never validates had a validation split
    carved out of its training data on resume that the first segment never had.
    The same run, silently trained on less."""
    import inspect

    from lamplighter.backend import runner as runner_mod

    source = inspect.getsource(runner_mod.RunManager.resume)
    assert "has_val=recipe.has_val" in source, "resume must pass the recipe's contract through"


def test_resume_sizes_the_lr_schedule_to_the_whole_plan():
    """A schedule built for `remaining` restarts its shape: resuming a cosine
    anneal at 20/40 blasts the model with peak LR again at epoch 21."""
    g, ns = _overfit_graph()
    mgr = _trained(g, ns)
    checkpoints.save("sched", manager=mgr)
    trained = len(mgr.history["train_loss"])

    mgr2 = RunManager()
    _resume(mgr2, "sched", epochs=trained + 6, namespace=ns)
    source = mgr2.snapshot["sources"]["trainer"]
    assert f"epochs={trained + 6}" in source, "the plan, not the remainder"
    assert f"start_epoch={trained}" in source, "and where this segment picks up"


def test_an_interrupted_run_resumes_onto_the_same_lr_curve():
    """The strongest form of the claim: interrupt a 12-epoch cosine anneal at 6
    and resume it, and the learning rate over all 12 epochs must be identical to
    an uninterrupted run of the same plan. Previously the second segment
    restarted the anneal from its peak, blasting the model with the starting LR
    at epoch 7 and undoing training."""
    def project(**extra):
        return _mlp_graph(
            {"scheduler": "CosineAnnealingLR", "lr": 0.1, "optimizer": "SGD", "seed": 0, **extra},
            data={"val_split": 0.25},
        )

    uninterrupted = RunManager()
    assert uninterrupted.start(project(epochs=12), _ns(n=64), emit=lambda m: None) is None
    assert uninterrupted.join(JOIN_TIMEOUT)
    whole = [round(v, 6) for v in uninterrupted.history["lr"]]
    assert len(whole) == 12

    # The same plan, stopped halfway — an interrupted run, not a shorter one.
    first = RunManager()
    halted = []

    def halt(message):
        if message.get("type") == "run_epoch" and message["epoch"] >= 6 and not halted:
            halted.append(1)
            first.stop()

    assert first.start(project(epochs=12), _ns(n=64), emit=halt) is None
    assert first.join(JOIN_TIMEOUT)
    checkpoints.save("interrupted", manager=first)

    second = RunManager()
    _resume(second, "interrupted", epochs=12, namespace=_ns(n=64))
    assert [round(v, 6) for v in second.history["lr"]] == whole


def test_a_fresh_run_emits_no_resume_machinery():
    """The positioning only exists on a resumed segment — a fresh run's source
    stays exactly what it always was."""
    from lamplighter.backend.codegen import generate_training

    g = _mlp_graph({"scheduler": "CosineAnnealingLR", "epochs": 10})
    src = generate_training(g.models[0].graph, g.training)
    assert "import warnings" not in src
    assert "catch_warnings" not in src
    assert "start_epoch=0" in src
