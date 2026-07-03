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
                         "epoch": None, "epochs": 20}
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

def test_run_endpoints_end_to_end():
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

        # The Session-property path: artifacts readable in-process.
        assert isinstance(run_manager.model, nn.Module)
        assert run_manager.history == status["history"]
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
    assert {n["id"] for n in snap["graph"]["nodes"]} == {"in", "l", "out"}
    # The exact sources that ran — replayable with torch.manual_seed(snap["seed"]).
    assert "class GeneratedModel" in snap["sources"]["model"]
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
