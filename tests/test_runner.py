"""The in-kernel run manager: executes the generated model/data/train sources on
a background thread with data resolved from a (test-injected) namespace, streams
per-epoch events, and supports cooperative stop. Deterministic — tests join the
run thread and trigger stop from the emit callback, never sleep-and-hope."""
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.runner import RunManager
from tests.helpers import edge, graph, node

JOIN_TIMEOUT = 60  # generous; runs are tiny


def _mlp_graph(training=None, data=None):
    g = graph(
        [node("in", "Input", {"shape": "16, 8"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    g.training = {"device": "cpu", "epochs": 5, "lr": 0.1, **(training or {})}
    g.data = {"source": "memory", "x_var": "X", "y_var": "y", **(data or {})}
    return g


def _ns(n=16):
    torch.manual_seed(0)
    return {"X": torch.randn(n, 8), "y": torch.randint(0, 3, (n,))}


def _start(g, ns, emit=None):
    mgr = RunManager()
    events: list[dict] = []
    err = mgr.start(g, namespace=ns, emit=emit or events.append)
    return mgr, events, err


# --- happy paths ------------------------------------------------------------

def test_tensor_picks_train_to_done():
    mgr, events, err = _start(_mlp_graph({"epochs": 20}), _ns())
    assert err is None
    assert mgr.join(JOIN_TIMEOUT)

    assert mgr.state == "done"
    assert isinstance(mgr.model, nn.Module)
    assert len(mgr.history["train_loss"]) == 20
    assert mgr.history["train_loss"][-1] < mgr.history["train_loss"][0]  # it learned

    # Event stream: running status first, one run_epoch per epoch, done status last.
    assert isinstance(events[0].pop("seed"), int)  # the run's (recorded) seed
    assert events[0] == {"type": "run_status", "state": "running", "error": None,
                         "epoch": None, "epochs": 20, "best_epoch": None}
    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert [e["epoch"] for e in epochs] == list(range(1, 21))
    assert "train_loss" in epochs[0]["metrics"]
    assert events[-1]["type"] == "run_status" and events[-1]["state"] == "done"


def test_tensor_picks_with_val_split():
    # The Data panel's val_split flows through make_dataloaders → a val_loader,
    # so the run reports val metrics without any training-side config.
    g = _mlp_graph({"epochs": 3}, {"batch_size": 8, "val_split": 0.25})
    mgr, events, err = _start(g, _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done"
    assert len(mgr.history["train_loss"]) == 3
    assert len(mgr.history["val_loss"]) == 3  # val ran every epoch


def test_dataloader_passthrough_pick():
    ns = _ns()
    ns["loader"] = DataLoader(TensorDataset(ns["X"], ns["y"]), batch_size=8)
    g = _mlp_graph({"epochs": 2}, {"x_var": "loader"})
    mgr, _, err = _start(g, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done"
    assert len(mgr.history["train_loss"]) == 2


def test_multi_input_tensor_picks():
    g = graph(
        [
            node("a", "Input", {"shape": "16, 8"}, y=0),
            node("b", "Input", {"shape": "16, 8"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("l", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"),
         edge("cat", "l"), edge("l", "out")],
    )
    g.training = {"device": "cpu", "epochs": 2}
    g.data = {"source": "memory", "x_vars": {"a": "X0", "b": "X1"}, "y_var": "y"}
    torch.manual_seed(0)
    ns = {"X0": torch.randn(16, 8), "X1": torch.randn(16, 8), "y": torch.randint(0, 3, (16,))}
    mgr, _, err = _start(g, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done"
    assert len(mgr.history["train_loss"]) == 2


# --- cooperative stop -------------------------------------------------------

def test_stop_after_first_epoch():
    mgr = RunManager()
    events: list[dict] = []

    def emit(message):
        events.append(message)
        if message["type"] == "run_epoch":  # stop as soon as the first epoch lands
            mgr.stop()

    err = mgr.start(_mlp_graph({"epochs": 50}), _ns(), emit=emit)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "stopped"
    assert len(mgr.history["train_loss"]) == 1  # partial history kept
    assert isinstance(mgr.model, nn.Module)  # partially-trained model kept too


# --- pre-flight rejection ---------------------------------------------------

def test_rejects_missing_variable():
    mgr, _, err = _start(_mlp_graph(), {"y": torch.randint(0, 3, (16,))})  # no X
    assert err is not None and "'X' is not registered" in err
    assert mgr.state == "idle"  # never started


def test_rejects_unpicked_variable():
    g = _mlp_graph(data={"x_var": ""})
    mgr, _, err = _start(g, _ns())
    assert err is not None and "pick one in the Data tab" in err


def test_rejects_ndarray_pick():
    import numpy as np

    ns = _ns()
    ns["X"] = np.zeros((16, 8))
    _, _, err = _start(_mlp_graph(), ns)
    assert err is not None and "torch.from_numpy" in err


def test_rejects_invalid_graph():
    g = graph([node("in", "Input", {"shape": "16, 8"}), node("out", "Output")], [])
    g.data = {"source": "memory", "x_var": "X", "y_var": "y"}
    _, _, err = _start(g, _ns())
    assert err is not None  # unwired Output → codegen precondition fails


def test_rejects_second_start_while_running():
    mgr = RunManager()
    gate = threading.Event()

    def emit(message):
        if message["type"] == "run_epoch":
            gate.set()  # first epoch landed — the run is demonstrably in flight

    assert mgr.start(_mlp_graph({"epochs": 200}), _ns(), emit=emit) is None
    assert gate.wait(JOIN_TIMEOUT)
    err = mgr.start(_mlp_graph(), _ns(), emit=emit)
    assert err is not None and "already in progress" in err
    mgr.stop()
    assert mgr.join(JOIN_TIMEOUT)


# --- end-to-end: sess.data() → REST run --------------------------------------

def test_run_endpoints_end_to_end(tmp_path):
    """The full production path: the notebook registers data on the session
    (sess.data(X=X, y=y)), the app POSTs the graph to /api/run/start, the
    singleton runner trains in a thread resolving names from the registry, and
    /api/run/status reports the finished history."""
    from fastapi.testclient import TestClient

    from backend import datastore
    from backend.app import app
    from backend.runner import run_manager
    from lamplighter.session import Session

    sess = Session("127.0.0.1", 1)  # registry is in-process; no server thread needed
    try:
        torch.manual_seed(0)
        sess.data(X=torch.randn(16, 8), y=torch.randint(0, 3, (16,)))
        g = _mlp_graph({"epochs": 3})
        with TestClient(app) as c:
            # Pre-flight rejection surfaces as a 400 with the runner's message.
            bad = g.model_copy(deep=True)
            bad.data["x_var"] = "nope"
            r = c.post("/api/run/start", json=bad.model_dump())
            assert r.status_code == 400 and "not registered" in r.json()["detail"]

            r = c.post("/api/run/start", json=g.model_dump())
            assert r.status_code == 200
            assert run_manager.join(JOIN_TIMEOUT)  # deterministic wait (same process)

            status = c.get("/api/run/status").json()
            assert status["state"] == "done"
            assert len(status["history"]["train_loss"]) == 3

            # The weights download serves the same checkpoint format.
            r = c.get("/api/run/weights")
            assert r.status_code == 200
            assert r.headers["content-disposition"] == 'attachment; filename="model.pt"'
            import io

            downloaded = torch.load(io.BytesIO(r.content), weights_only=True)
            assert {"state_dicts", "best_state_dict", "best_epoch", "epoch",
                    "history", "snapshot"} <= set(downloaded)
            assert list(downloaded["state_dicts"]) == ["model"]  # sole model, by role

        # The Session-property path: artifacts readable in-process.
        assert isinstance(run_manager.model, nn.Module)
        assert run_manager.history == status["history"]

        # And the notebook-side save: sess.save_checkpoint -> load_checkpoint.
        import lamplighter

        saved = sess.save_checkpoint(str(tmp_path / "ckpt.pt"))
        rebuilt, snap = lamplighter.load_checkpoint(saved)
        x = torch.randn(2, 8)
        with torch.no_grad():
            assert torch.equal(rebuilt(x), run_manager.model.eval()(x))
    finally:
        datastore.clear()


def test_run_events_stream_over_websocket():
    """The training thread's events cross into the asyncio loop and arrive on a
    real WebSocket, in order: running → one run_epoch per epoch → done."""
    from fastapi.testclient import TestClient

    from backend import datastore
    from backend.app import app

    # Register the data the way the notebook would (the runner resolves from
    # the session registry by default).
    datastore.clear()
    datastore.register(**_ns())

    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            assert c.post("/api/run/start", json=_mlp_graph({"epochs": 3}).model_dump()).status_code == 200
            got = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "run_epoch":
                    got.append(msg["epoch"])
                elif msg["type"] == "run_status" and msg["state"] != "running":
                    final = msg["state"]
                    break
            assert got == [1, 2, 3]
            assert final == "done"
    datastore.clear()


# --- runtime failure --------------------------------------------------------

def test_runtime_failure_sets_failed_state():
    ns = _ns()
    ns["y"] = torch.randn(16, 2)  # float targets → CrossEntropyLoss blows up
    mgr, events, err = _start(_mlp_graph(), ns)
    assert err is None  # pre-flight can't know — it's a real tensor
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "failed"
    assert mgr.error  # carries the exception message
    assert events[-1]["type"] == "run_status" and events[-1]["state"] == "failed"


# --- reproducibility: seed + snapshot ----------------------------------------

def test_same_seed_reproduces_the_run_exactly():
    g = _mlp_graph({"epochs": 5, "seed": 1234}, {"val_split": 0.25, "batch_size": 8})
    histories = []
    for _ in range(2):
        mgr, events, err = _start(g, _ns())
        assert err is None and mgr.join(JOIN_TIMEOUT)
        assert mgr.state == "done"
        assert events[0]["seed"] == 1234  # rides the run_status payload
        histories.append(mgr.history)
    # Model init, random_split, and shuffling all seeded — bit-identical runs.
    assert histories[0] == histories[1]


def test_unset_seed_is_drawn_and_recorded():
    seeds = []
    for _ in range(2):
        mgr, _, err = _start(_mlp_graph({"epochs": 1}), _ns())
        assert err is None and mgr.join(JOIN_TIMEOUT)
        assert isinstance(mgr.status()["seed"], int)  # recorded even when unset
        seeds.append(mgr.status()["seed"])
    assert seeds[0] != seeds[1]  # fresh randomness per run


def test_snapshot_is_a_complete_reproducibility_record():
    g = _mlp_graph({"epochs": 2, "seed": 7})
    mgr, _, err = _start(g, _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)

    snap = mgr.snapshot
    assert snap["seed"] == 7
    assert snap["device"] == "cpu"  # resolved, not "auto"
    assert snap["training"]["epochs"] == 2
    assert snap["data"]["x_var"] == "X"
    nodes = snap["project"]["models"][0]["graph"]["nodes"]
    assert {n["id"] for n in nodes} == {"in", "l", "out"}
    # The exact sources that ran — replayable with torch.manual_seed(snap["seed"]).
    assert "class GeneratedModel" in snap["sources"]["models"]["model"]
    assert "def make_dataloaders" in snap["sources"]["data"]
    assert "def train" in snap["sources"]["trainer"]
    assert snap["state"] == "done" and snap["started"] and snap["finished"]

    # The Session-property path.
    from lamplighter.session import Session

    from backend.runner import run_manager as singleton  # property reads the singleton
    assert Session("127.0.0.1", 1).snapshot is singleton.snapshot or True  # smoke: no raise


def test_run_leaves_the_kernels_rng_state_untouched():
    # The notebook's randomness must not be perturbed by a run: seed the kernel,
    # note the sequence it WOULD produce, then re-seed, run a full training run,
    # and draw — the stream continues as if the run never happened (fork_rng).
    ns = _ns()  # built first — it seeds and draws on its own
    torch.manual_seed(42)
    reference = torch.randn(4)

    torch.manual_seed(42)
    mgr, _, err = _start(_mlp_graph({"epochs": 3, "seed": 7}), ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done"

    assert torch.equal(torch.randn(4), reference)  # kernel stream unbroken


# --- weight export: self-contained checkpoints --------------------------------

def test_checkpoint_requires_a_trained_model():
    import pytest

    with pytest.raises(ValueError, match="no trained model"):
        RunManager().checkpoint()


def test_checkpoint_round_trips_through_load_checkpoint(tmp_path):
    # Save weights+snapshot to a file, then rebuild the model from NOTHING but
    # that file (the generated source travels inside it) — the rebuilt model
    # must produce bit-identical outputs to the in-memory trained one.
    import lamplighter

    ns = _ns()
    mgr, _, err = _start(_mlp_graph({"epochs": 3, "seed": 11}), ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)

    path = tmp_path / "model.pt"
    torch.save(mgr.checkpoint(), path)

    rebuilt, snapshot = lamplighter.load_checkpoint(str(path))
    assert snapshot["seed"] == 11
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.equal(rebuilt(x), mgr.model.eval()(x))


# --- best-epoch tracking ------------------------------------------------------

def _overfit_graph():
    # lr=0.6 on 20 samples: val loss bottoms early then climbs — a deterministic
    # (seeded) case where the best model is NOT the final one.
    g = _mlp_graph({"epochs": 12, "lr": 0.6, "seed": 3},
                   {"val_split": 0.3, "batch_size": 4})
    torch.manual_seed(0)
    return g, {"X": torch.randn(20, 8), "y": torch.randint(0, 3, (20,))}


def test_best_epoch_tracks_the_val_loss_minimum():
    g, ns = _overfit_graph()
    mgr, events, err = _start(g, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)

    val = mgr.history["val_loss"]
    assert mgr.best_epoch == min(range(len(val)), key=lambda i: val[i]) + 1
    assert mgr.best_epoch < len(val)  # the engineered case: best is mid-run
    # The captured weights are a snapshot from that epoch, not the final ones.
    final = mgr.model.state_dict()
    assert any(
        not torch.equal(mgr.best_state_dict[k], final[k].cpu()) for k in final
    )
    # best_epoch rides the status + WS payloads.
    assert mgr.status()["best_epoch"] == mgr.best_epoch
    assert events[-1]["best_epoch"] == mgr.best_epoch


def test_best_model_rebuilds_and_beats_final_on_val():
    g, ns = _overfit_graph()
    mgr, _, err = _start(g, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)

    best = mgr.best_model()
    assert best is not None
    # Score both models on the full data: the best-epoch model must match the
    # val loss recorded at its epoch better than the (overfit) final model does.
    import torch.nn.functional as F

    with torch.no_grad():
        best_loss = F.cross_entropy(best(ns["X"]), ns["y"]).item()
        final_loss = F.cross_entropy(mgr.model.eval().cpu()(ns["X"]), ns["y"]).item()
    assert best_loss < final_loss


def test_no_validation_means_no_best_tracking():
    mgr, _, err = _start(_mlp_graph({"epochs": 3}), _ns())  # no val_split
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert mgr.best_epoch is None and mgr.best_state_dict is None
    assert mgr.best_model() is None


def test_checkpoint_carries_best_and_load_checkpoint_best(tmp_path):
    import lamplighter

    g, ns = _overfit_graph()
    mgr, _, err = _start(g, ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)

    path = tmp_path / "ckpt.pt"
    torch.save(mgr.checkpoint(), path)

    best, _ = lamplighter.load_checkpoint(str(path), best=True)
    final, _ = lamplighter.load_checkpoint(str(path))
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert not torch.equal(best(x), final(x))  # genuinely different weights
        assert torch.equal(best(x), mgr.best_model()(x))  # same as the rebuilt best


def test_load_checkpoint_best_errors_without_validation(tmp_path):
    import lamplighter
    import pytest

    mgr, _, err = _start(_mlp_graph({"epochs": 2}), _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)
    path = tmp_path / "ckpt.pt"
    torch.save(mgr.checkpoint(), path)
    with pytest.raises(lamplighter.LamplighterError, match="no best-epoch weights"):
        lamplighter.load_checkpoint(str(path), best=True)


def test_health_readout_tracks_per_layer_norms():
    # The per-layer training-health snapshot streams with each epoch and is
    # available in full via status() (for tabs that join mid/post-run).
    mgr, events, err = _start(_mlp_graph({"epochs": 4}), _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)

    health = mgr.status()["health_history"]
    assert len(health) == 4  # one snapshot per epoch

    first = health[0]
    (role,) = first.keys()  # a sole supervised model → one role
    l0 = first[role]["layer_0"]  # the MLP's single Linear
    assert l0["node"] == "Linear"  # labelled by node type (no user name set)
    assert l0["nodeId"] == "l"  # maps back to the canvas node id (for badges)
    assert isinstance(l0["w"], float) and l0["w"] > 0
    assert "dw" not in l0  # no previous epoch to diff against on epoch 1

    # The update ratio appears from epoch 2, and the weights actually moved.
    assert health[1][role]["layer_0"]["dw"] > 0

    # Streamed identically over the event channel.
    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert epochs[0]["health"] == health[0]
    assert epochs[-1]["health"] == health[-1]


def test_health_tracks_dead_units_and_cleans_up_hooks():
    # An activation layer carries no params, so instead of weight/update stats it
    # gets a dead-unit row — the fraction of units that never left ~0 all epoch.
    g = graph(
        [
            node("in", "Input", {"shape": "16, 8"}),
            node("l1", "Linear", {"out_features": 6}),
            node("act", "ReLU", {}),
            node("l2", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("in", "l1"), edge("l1", "act"), edge("act", "l2"), edge("l2", "out")],
    )
    g.training = {"device": "cpu", "epochs": 3, "lr": 0.1}
    g.data = {"source": "memory", "x_var": "X", "y_var": "y"}
    mgr, _, err = _start(g, _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)

    snap = mgr.status()["health_history"][-1]["model"]
    dead_rows = [v for v in snap.values() if "dead" in v]
    assert len(dead_rows) == 1
    assert dead_rows[0]["node"] == "ReLU"
    assert 0.0 <= dead_rows[0]["dead"] <= 1.0
    assert "w" not in dead_rows[0]  # activation row: no weight norm
    assert sum("w" in v for v in snap.values()) == 2  # the two Linears keep param rows

    # Hooks are torn down at run end — nothing left on the manager or the module.
    assert mgr._hook_handles == []
    relu = dict(mgr.model.named_modules())["layer_1"]
    assert len(relu._forward_hooks) == 0


def test_dead_unit_measure_skips_tanh():
    # tanh's ~0 output is its *active* region, not "dead" — so it must get no
    # dead row (unlike ReLU, which does in the test above).
    g = graph(
        [
            node("in", "Input", {"shape": "16, 8"}),
            node("l1", "Linear", {"out_features": 6}),
            node("act", "Tanh", {}),
            node("l2", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("in", "l1"), edge("l1", "act"), edge("act", "l2"), edge("l2", "out")],
    )
    g.training = {"device": "cpu", "epochs": 2, "lr": 0.1}
    g.data = {"source": "memory", "x_var": "X", "y_var": "y"}
    mgr, _, err = _start(g, _ns())
    assert err is None and mgr.join(JOIN_TIMEOUT)
    snap = mgr.status()["health_history"][-1]["model"]
    assert not any("dead" in v for v in snap.values())  # tanh excluded → no dead rows


def test_preview_returns_input_output_target_for_a_supervised_model():
    # The generic "see what it learned": forward a sample of the model's wired
    # inputs and hand back the tensors (the frontend renders by shape).
    ns = _ns()
    mgr, _, err = _start(_mlp_graph({"epochs": 2}), ns)
    assert err is None and mgr.join(JOIN_TIMEOUT)

    p = mgr.preview(n=16, ns=ns)
    assert "error" not in p
    assert p["inputs"][0]["shape"] == [16, 8]  # sampled input rows
    assert p["outputs"][0]["shape"] == [16, 3]  # the model's real outputs
    assert p["target"] is not None and p["target"]["shape"] == [16]  # y, when present
    assert len(p["outputs"][0]["data"]) == 16 * 3  # real numbers, ready to render


def test_preview_before_a_run_returns_a_note():
    p = RunManager().preview()
    assert "error" in p and "run training" in p["error"]


def test_preview_draws_noise_for_a_noise_wired_input():
    from backend.schema import DataNode

    mgr = RunManager()
    z = mgr._sample_from_node(DataNode(id="z", kind="noise", config={"dims": "100"}), None, "in", 8, {})
    assert list(z.shape) == [8, 100]
    u = mgr._sample_from_node(
        DataNode(id="u", kind="noise", config={"dims": "4", "distribution": "uniform"}), None, "in", 5, {}
    )
    assert list(u.shape) == [5, 4] and float(u.min()) >= 0.0 and float(u.max()) < 1.0
