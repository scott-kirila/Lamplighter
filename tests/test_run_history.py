"""First-class runs: every terminal run auto-records (weightless) into the run
store under a reserved run-N name; retention prunes only unnamed weightless
records (failed first); view stays read-only; weights-requiring actions refuse
weightless runs."""
import pytest

from lamplighter.backend import checkpoints
from lamplighter.backend.runner import RunManager
from tests.test_runner import JOIN_TIMEOUT, _mlp_graph, _ns


@pytest.fixture(autouse=True)
def _clean_store():
    checkpoints.clear()
    yield
    checkpoints.clear()


def _recording_run(training=None, ns=None):
    """A record_runs manager that finished a run (auto-recording like the
    production singleton does)."""
    mgr = RunManager(record_runs=True)
    events: list = []
    err = mgr.start(_mlp_graph(training or {"epochs": 2}), namespace=ns or _ns(), emit=events.append)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    return mgr, events


def _weightless(name, state="done", created="2026-01-01T00:00:00", study=None):
    """Inject a minimal auto record directly (unit-scale; no training)."""
    snapshot = {"state": state, "training": {"epochs": 1}, "seed": 1}
    if study is not None:
        snapshot["study"] = study
    checkpoints._store[name] = {
        "checkpoint": {
            "state_dicts": None,
            "best_state_dict": None,
            "best_epoch": None,
            "epoch": 1,
            "history": {"train_loss": [1.0]},
            "health_history": [],
            "steps": [],
            "step_total": 0,
            "snapshot": snapshot,
        },
        "created": created,
        "auto": True,
    }


def test_terminal_runs_auto_record_weightless_under_their_reserved_name():
    mgr, events = _recording_run()
    assert mgr.run_name and mgr.run_name.startswith("run-")
    # The running status already carried the name, so the list can show the
    # live run before its record exists.
    running = next(e for e in events if e.get("type") == "run_status" and e["state"] == "running")
    assert running["run_name"] == mgr.run_name

    (meta,) = [m for m in checkpoints.metas() if m["name"] == mgr.run_name]
    assert meta["has_weights"] is False
    assert meta["auto"] is True
    assert meta["state"] == "done"
    assert meta["source"] == "app"
    assert meta["epoch"] == 2 and meta["epochs"] == 2
    # The record is the full run: curves, and its snapshot for reproducibility.
    rec = checkpoints.load(mgr.run_name)
    assert rec["state_dicts"] is None
    assert len(rec["history"]["train_loss"]) == 2
    assert rec["snapshot"]["seed"] == mgr.seed


def test_failed_runs_record_too():
    mgr = RunManager(record_runs=True)
    ns = {"X": _ns()["X"], "y": _ns()["y"][:3]}  # misaligned → the run fails
    err = mgr.start(_mlp_graph({"epochs": 1}), namespace=ns, emit=lambda m: None)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "failed"
    (meta,) = [m for m in checkpoints.metas() if m["name"] == mgr.run_name]
    assert meta["state"] == "failed" and meta["has_weights"] is False


def test_keeping_weights_upgrades_the_auto_record_in_place():
    mgr, _ = _recording_run()
    checkpoints.save(mgr.run_name, manager=mgr)  # the "keep weights" action
    (meta,) = [m for m in checkpoints.metas() if m["name"] == mgr.run_name]
    assert meta["has_weights"] is True
    assert meta["auto"] is False  # kept — exempt from retention


def test_restore_marks_the_restored_run_as_the_kernels_current():
    # Restoring a stored run makes it the kernel's current run, so status()
    # reports its name — the runs list marks that row as shown, surviving a
    # refresh (which rehydrates from status).
    mgr, _ = _recording_run()
    checkpoints.save(mgr.run_name, manager=mgr)
    name = mgr.run_name

    fresh = RunManager()
    assert fresh.run_name is None
    assert fresh.restore(checkpoints.load(name), name=name) is None
    assert fresh.run_name == name
    assert fresh.status()["run_name"] == name


def test_retention_prunes_oldest_weightless_autos_failed_first():
    for i in range(checkpoints._AUTO_KEEP):
        _weightless(f"run-{i}", created=f"2026-01-01T00:00:{i:02d}")
    _weightless("run-bad", state="failed", created="2026-01-01T00:00:30")  # newer than all
    checkpoints._store["kept"] = dict(checkpoints._store["run-0"])  # renamed → not auto
    checkpoints._store["kept"]["auto"] = False

    checkpoints._prune()
    # One over the cap: the failed record goes first despite being newest.
    assert "run-bad" not in checkpoints._store
    assert "run-0" in checkpoints._store and "kept" in checkpoints._store

    _weightless("run-new", created="2026-01-01T00:01:00")
    checkpoints._prune()
    assert "run-0" not in checkpoints._store  # now the oldest auto goes
    assert "kept" in checkpoints._store  # named entries never prune


def test_retention_pools_trials_and_regular_runs_separately():
    # A sweep's trials must not evict the training history (nor the reverse):
    # each pool keeps its own newest _AUTO_KEEP.
    for i in range(checkpoints._AUTO_KEEP):
        _weightless(f"run-{i}", created=f"2026-01-01T00:00:{i:02d}")
    for i in range(checkpoints._AUTO_KEEP + 2):
        _weightless(f"trial-{i}", created=f"2026-01-01T01:00:{i:02d}", study="s1")

    checkpoints._prune()
    names = set(checkpoints._store)
    # The trial pool is 2 over cap → its two OLDEST go; regular runs untouched
    # even though the store holds far more than one cap in total.
    assert "trial-0" not in names and "trial-1" not in names
    assert all(f"trial-{i}" in names for i in range(2, checkpoints._AUTO_KEEP + 2))
    assert all(f"run-{i}" in names for i in range(checkpoints._AUTO_KEEP))


def test_rename_clears_auto_and_keeps_listing_position():
    _weightless("run-1")
    _weightless("run-2")
    meta = checkpoints.rename("run-1", "good-one")
    assert meta["auto"] is False
    assert [m["name"] for m in checkpoints.metas()] == ["good-one", "run-2"]
    with pytest.raises(ValueError, match="already exists"):
        checkpoints.rename("run-2", "good-one")


def test_weightless_runs_refuse_restore_and_resume():
    _weightless("run-1")
    fresh = RunManager()
    err = fresh.restore(checkpoints.load("run-1"))
    assert err is not None and "kept no weights" in err
    err = fresh.resume("run-1", checkpoints.load("run-1"), epochs=5, namespace=_ns(), emit=lambda m: None)
    assert err is not None and "kept no weights" in err


def test_view_endpoint_is_read_only_and_status_shaped():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app
    from lamplighter.backend.runner import run_manager

    _weightless("run-1")
    before = run_manager.status()
    client = TestClient(app)
    res = client.get("/api/checkpoints/run-1/view")
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "done" and body["epochs"] == 1
    assert body["history"] == {"train_loss": [1.0]}
    assert body["config"]["epochs"] == 1
    assert run_manager.status() == before  # the kernel's run is untouched

    assert client.get("/api/checkpoints/run-1/weights").status_code == 409
    assert client.get("/api/checkpoints/missing/view").status_code == 404


def test_keep_weights_endpoint_refuses_a_run_the_kernel_no_longer_holds():
    """Keep-weights clones the LIVE model under the given name. After restoring
    an old run, the live model belongs to THAT run — keeping the newer, still
    weightless run under its own name would store the wrong weights. The
    endpoint refuses (409) rather than mislabel them; the run stays weightless."""
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app
    from lamplighter.backend.runner import run_manager
    from tests.test_runner import JOIN_TIMEOUT, _mlp_graph, _ns

    client = TestClient(app)

    # Run A on the kernel, then keep it — the kernel holds it, so this succeeds.
    assert run_manager.start(_mlp_graph({"epochs": 2}), namespace=_ns(), emit=lambda m: None) is None
    assert run_manager.join(JOIN_TIMEOUT)
    run_a = run_manager.run_name
    assert client.post("/api/checkpoints", json={"name": run_a}).status_code == 200

    # Run B — a newer run, left weightless.
    assert run_manager.start(_mlp_graph({"epochs": 2}), namespace=_ns(), emit=lambda m: None) is None
    assert run_manager.join(JOIN_TIMEOUT)
    run_b = run_manager.run_name
    assert run_b != run_a and checkpoints.load(run_b)["state_dicts"] is None

    # Restore run A — the live model (and run_name) become run A's again.
    assert client.post(f"/api/checkpoints/{run_a}/restore").status_code == 200
    assert run_manager.run_name == run_a

    # Keeping run B now would mislabel run A's weights → refused, B stays weightless.
    res = client.post("/api/checkpoints", json={"name": run_b})
    assert res.status_code == 409 and "no longer holds" in res.json()["detail"]
    assert checkpoints.load(run_b)["state_dicts"] is None


def test_preview_a_saved_run_rebuilds_its_weights_without_touching_the_kernel():
    """The Preview tab previews a stored run by name — rebuild its saved weights,
    forward a sample, hand back outputs — all without disturbing the live model,
    so you can flip between runs freely."""
    ns = _ns()
    mgr, _ = _recording_run(ns=ns)
    checkpoints.save(mgr.run_name, manager=mgr)  # keep weights
    live = mgr.models  # the live model dict — must survive the preview

    p = mgr.preview_checkpoint(checkpoints.load(mgr.run_name), ns=ns)
    assert "error" not in p
    # One output row per sampled input row, real numbers ready to render.
    assert p["outputs"][0]["shape"][0] == p["inputs"][0]["shape"][0]
    assert len(p["outputs"][0]["data"]) > 0
    assert mgr.models is live  # rebuilt fresh, local instances — kernel untouched


def test_preview_endpoint_refuses_a_weightless_run():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    _weightless("run-1")
    client = TestClient(app)
    res = client.get("/api/checkpoints/run-1/preview")
    assert res.status_code == 409 and "can't be previewed" in res.json()["detail"]
    assert client.get("/api/checkpoints/missing/preview").status_code == 404


def test_rename_endpoint():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    _weightless("run-1")
    client = TestClient(app)
    assert client.post("/api/checkpoints/run-1/rename", json={"name": "keeper"}).status_code == 200
    assert [m["name"] for m in checkpoints.metas()] == ["keeper"]
    assert client.post("/api/checkpoints/missing/rename", json={"name": "x"}).status_code == 404


def test_checkpoints_carry_the_format_version():
    # Every checkpoint shape the runner builds — the weightless auto record,
    # the kept-weights checkpoint, and (via save) the stored entry — carries
    # CHECKPOINT_VERSION, the hook future migrations key on.
    from lamplighter.backend.checkpoints import CHECKPOINT_VERSION

    mgr, _ = _recording_run()
    assert mgr.run_record()["version"] == CHECKPOINT_VERSION
    assert mgr.checkpoint()["version"] == CHECKPOINT_VERSION
    checkpoints.save(mgr.run_name, manager=mgr)
    assert checkpoints.load(mgr.run_name)["version"] == CHECKPOINT_VERSION


# --- run → model attribution (which run trained which model) -----------------

def _two_model_project(target="m2"):
    """Two independent Supervised MLPs sharing one training config; ``target`` is
    the model the ``model`` role points to (and the dataset feeds)."""
    from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project
    from tests.helpers import edge, graph, node

    def mlp():
        g = graph(
            [node("in", "Input", {"shape": "16, 8"}), node("l", "Linear", {"out_features": 3}),
             node("out", "Output")],
            [edge("in", "l"), edge("l", "out")],
        )
        return Graph(nodes=g.nodes, edges=g.edges)

    return Project(
        models=[ModelDef(id="m1", name="Alpha", graph=mlp()), ModelDef(id="m2", name="Beta", graph=mlp())],
        data_nodes=[DataNode(id="data", kind="dataset", name="Data",
                             config={"source": "memory", "x_var": "X", "y_var": "y"})],
        links=[ModelLink(id="L", source_data="data", target_model=target)],
        training={"recipe": "supervised", "device": "cpu", "epochs": 2, "lr": 0.1,
                  "roles": {"model": target}},
    )


def test_run_models_from_explicit_roles_auto_and_empty():
    from lamplighter.backend.checkpoints import run_models_from

    # Explicit roles (a GAN): both models, names from the snapshot's project.
    snap = {
        "training": {"roles": {"generator": "g", "discriminator": "d"}},
        "project": {"models": [{"id": "g", "name": "Gen"}, {"id": "d", "name": "Disc"}]},
    }
    assert run_models_from(snap) == [
        {"role": "generator", "id": "g", "name": "Gen"},
        {"role": "discriminator", "id": "d", "name": "Disc"},
    ]
    # Auto-assigned single model (roles empty): the sole model is the target.
    snap1 = {"training": {}, "project": {"models": [{"id": "model", "name": "Net"}]}}
    assert run_models_from(snap1) == [{"role": "model", "id": "model", "name": "Net"}]
    # Multi-model with no roles recorded → not attributable; no snapshot → [].
    assert run_models_from({"training": {}, "project": {"models": [{"id": "a"}, {"id": "b"}]}}) == []
    assert run_models_from(None) == []


def test_run_models_from_name_is_frozen_at_run_time():
    # The name comes from the snapshot's OWN project dump, so a run stays
    # attributed to the name it trained under even after a later rename/delete.
    from lamplighter.backend.checkpoints import run_models_from

    snap = {"training": {"roles": {"model": "m1"}},
            "project": {"models": [{"id": "m1", "name": "OldName"}]}}
    assert run_models_from(snap) == [{"role": "model", "id": "m1", "name": "OldName"}]


def test_auto_single_model_run_attributes_to_its_sole_model():
    mgr, _ = _recording_run()
    (meta,) = [m for m in checkpoints.metas() if m["name"] == mgr.run_name]
    assert meta["models"] == [{"role": "model", "id": "model", "name": "Model"}]


def test_supervised_run_attributes_to_the_targeted_model():
    # Two models, the role targets Beta (m2) — the recorded run names Beta, not
    # Alpha, so the Runs list can scope/label it correctly.
    mgr = RunManager(record_runs=True)
    assert mgr.start(_two_model_project(target="m2"), namespace=_ns(), emit=lambda m: None) is None
    assert mgr.join(JOIN_TIMEOUT)
    (meta,) = [m for m in checkpoints.metas() if m["name"] == mgr.run_name]
    assert meta["models"] == [{"role": "model", "id": "m2", "name": "Beta"}]
