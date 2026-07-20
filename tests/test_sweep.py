"""The Optimize engine (Phase A, headless): a sweep is N sequential trials,
each a REAL managed run with the trial's params merged into project.training —
recorded under the sweep's study tag, prunable via the cooperative stop, the
best trial's weights kept as they happen and renamed <study>-best at the end."""
import optuna
import pytest
import torch

from lamplighter.backend import checkpoints, datastore
from lamplighter.backend.runner import RunManager
from lamplighter.backend.sweep import SweepManager, _check_config
from tests.test_runner import JOIN_TIMEOUT, _mlp_graph


@pytest.fixture(autouse=True)
def _clean_state():
    checkpoints.clear()
    datastore.clear()
    torch.manual_seed(0)
    datastore.register(X=torch.randn(32, 8), y=torch.randint(0, 3, (32,)))
    yield
    checkpoints.clear()
    datastore.clear()


def _project(**training):
    # seed pinned: trial TRAINING seeds otherwise draw from python's global
    # random, making curves — and pruner decisions — depend on suite order.
    return _mlp_graph(
        {"epochs": 3, "device": "cpu", "lr": 0.1, "seed": 0, **training},
        data={"val_split": 0.25},
    )


def _sweep(events=None):
    """A SweepManager over its own recording RunManager, events captured."""
    sink = events if events is not None else []
    return SweepManager(manager=RunManager(record_runs=True), emit=sink.append), sink


def _config(**over):
    return {
        "study": "s1",
        "n_trials": 3,
        "seed": 7,  # seeds the SAMPLER — the sweep itself is reproducible
        "params": [
            {"name": "lr", "type": "float", "low": 1e-3, "high": 1e-1, "log": True},
            {"name": "optimizer", "type": "categorical", "choices": ["Adam", "SGD"]},
        ],
        **over,
    }


# --- the full loop -----------------------------------------------------------

def test_sweep_runs_trials_as_recorded_runs_and_keeps_the_best():
    # prune=False: this is the happy-path loop — a (legitimate) median-prune of
    # a bad-lr trial would just add noise here; pruning has its own test.
    sweep, _ = _sweep()
    assert sweep.start(_project(), _config(prune=False)) is None
    assert sweep.join(JOIN_TIMEOUT * 3)
    assert sweep.state == "done", sweep.error

    # Every trial landed in the run store, tagged with its study and source.
    trials = [m for m in checkpoints.metas() if m["study"] == "s1"]
    assert len(trials) == 3
    snaps = [checkpoints.load(m["name"])["snapshot"] for m in trials]
    assert all(s["source"] == "sweep" and s["study"] == "s1" for s in snaps)

    # Each trial's snapshot shows ITS OWN suggested params — distinct lrs, and
    # the optimizer choice recorded in the training config it actually ran.
    lrs = [s["training"]["lr"] for s in snaps]
    assert len(set(lrs)) == 3
    assert all(s["training"]["optimizer"] in ("Adam", "SGD") for s in snaps)

    # The best trial: value == the minimum final val_loss across trials, its
    # weights were kept (restorable), and it wears the sweep's artifact name.
    finals = {m["name"]: checkpoints.load(m["name"])["history"]["val_loss"][-1] for m in trials}
    assert sweep.best is not None
    assert sweep.best["run_name"] == "s1-best"
    assert sweep.best["value"] == min(finals.values())
    best = checkpoints.load("s1-best")
    assert best["state_dicts"] is not None  # weights kept as it happened
    (best_meta,) = [m for m in checkpoints.metas() if m["name"] == "s1-best"]
    assert best_meta["auto"] is False  # named + saved → exempt from retention
    # The best's recorded params match its own snapshot.
    assert sweep.best["params"]["lr"] == checkpoints.load("s1-best")["snapshot"]["training"]["lr"]

    status = sweep.status()
    assert status["completed"] == 3 and status["pruned"] == 0 and status["failed"] == 0


def test_pruned_trials_record_as_stopped_runs():
    # A threshold pruner that every report trips (val_loss > 1e-9 always) —
    # deterministic pruning through the REAL plumbing: report → should_prune →
    # cooperative stop → the run records as "stopped".
    sweep, _ = _sweep()
    pruner = optuna.pruners.ThresholdPruner(upper=1e-9)
    assert sweep.start(_project(), _config(n_trials=2), pruner=pruner) is None
    assert sweep.join(JOIN_TIMEOUT * 2)
    assert sweep.state == "done", sweep.error

    assert sweep.pruned == 2 and sweep.completed == 0
    assert sweep.best is None  # nothing completed — no artifact to crown
    trials = [m for m in checkpoints.metas() if m["study"] == "s1"]
    assert len(trials) == 2
    assert all(m["state"] == "stopped" for m in trials)


def test_missing_metric_fails_the_sweep_fast_with_the_fix():
    # No validation split → no val_loss → every trial would fail identically,
    # so the sweep aborts after the first with the fix spelled out.
    sweep, _ = _sweep()
    project = _mlp_graph({"epochs": 2, "device": "cpu"}, data={})  # no val_split
    assert sweep.start(project, _config(n_trials=3)) is None
    assert sweep.join(JOIN_TIMEOUT)
    assert sweep.state == "failed"
    assert "val_loss" in sweep.error and "validation split" in sweep.error


def test_stop_ends_the_sweep_between_trials():
    events: list = []
    sweep, _ = _sweep(events)

    # Stop the moment the first trial's terminal status lands — the loop must
    # not start trial 2.
    def emit(msg):
        events.append(msg)
        if msg.get("type") == "sweep_status" and msg.get("state") == "running" and (
            msg.get("completed", 0) + msg.get("pruned", 0) + msg.get("failed", 0)
        ) == 1:
            sweep.stop()

    sweep._emit_override = emit
    assert sweep.start(_project(), _config(n_trials=5)) is None
    assert sweep.join(JOIN_TIMEOUT * 2)
    assert sweep.state == "stopped"
    assert sweep.completed + sweep.pruned + sweep.failed == 1
    assert len([m for m in checkpoints.metas() if m["study"] == "s1"]) == 1


def test_trials_stream_run_events_to_the_normal_sink():
    # The composed emit forwards run_status/run_epoch — open tabs watch trials
    # stream exactly like hand-started runs.
    sweep, events = _sweep()
    assert sweep.start(_project(), _config(n_trials=1)) is None
    assert sweep.join(JOIN_TIMEOUT)
    types = {e.get("type") for e in events}
    assert {"sweep_status", "run_status", "run_epoch"} <= types


# --- guards and validation ---------------------------------------------------

def test_config_validation_speaks_user():
    assert "at least one hyperparameter" in _check_config({"params": [], "n_trials": 3})
    assert "unknown type" in _check_config(
        {"params": [{"name": "lr", "type": "bogus"}], "n_trials": 3})
    assert "needs choices" in _check_config(
        {"params": [{"name": "opt", "type": "categorical"}], "n_trials": 3})
    assert "needs low and high" in _check_config(
        {"params": [{"name": "lr", "type": "float", "low": 0.1}], "n_trials": 3})
    assert "n_trials" in _check_config(
        {"params": [{"name": "lr", "type": "float", "low": 0.0, "high": 1.0}], "n_trials": 0})
    assert _check_config(_config()) is None


def test_sweep_refuses_while_a_run_or_sweep_is_live():
    busy = RunManager()
    busy.state = "running"
    sweep = SweepManager(manager=busy, emit=lambda m: None)
    assert "run is in progress" in sweep.start(_project(), _config())

    sweep2, _ = _sweep()
    sweep2.state = "running"
    assert "already in progress" in sweep2.start(_project(), _config())


def test_start_refusal_aborts_the_sweep_with_the_reason():
    # An unrunnable project (no data registered) fails every trial identically —
    # the sweep must abort with the runner's message, not grind through n_trials.
    datastore.clear()  # nothing registered → the pick can't resolve
    sweep, _ = _sweep()
    assert sweep.start(_project(), _config(n_trials=4)) is None
    assert sweep.join(JOIN_TIMEOUT)
    assert sweep.state == "failed"
    assert "not registered" in sweep.error


def test_missing_optuna_surfaces_the_install_hint(monkeypatch):
    # Blocking the import (None in sys.modules makes `import optuna` raise)
    # must yield the exact user-facing hint, not a raw ImportError.
    import sys

    monkeypatch.setitem(sys.modules, "optuna", None)
    sweep, _ = _sweep()
    err = sweep.start(_project(), _config())
    assert err is not None and 'pip install "lamplighter[sweep]"' in err


def test_sweep_endpoints_are_wired():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    with TestClient(app) as c:
        # Bad config → 400 with the user-facing reason.
        res = c.post("/api/sweep/start", json={"project": _project().model_dump(),
                                               "config": {"params": [], "n_trials": 3}})
        assert res.status_code == 400
        assert "at least one hyperparameter" in res.json()["detail"]
        # Status hydration shape.
        status = c.get("/api/sweep/status").json()
        assert status["state"] == "idle" and status["best"] is None
        assert c.post("/api/sweep/stop").json()["ok"] is True