"""What a run tells you when it goes wrong.

Two failure modes that used to be silent: a loss that diverges to NaN (the run
kept reporting "done" while the dashboard quietly froze at the last finite
epoch), and an exception inside generated code (reduced to one line, discarding
the frames that ``exec_generated``'s linecache registration exists to make
readable).
"""
import json
import math

import torch

from lamplighter.backend.runner import RunManager, _finite
from tests.helpers import edge, graph, node, single_model_project

JOIN_TIMEOUT = 60


def _mlp_graph(training=None, data=None):
    g = graph(
        [node("in", "Input", {"shape": "1, 8"}),
         node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return single_model_project(
        g,
        training={"device": "cpu", "epochs": 3, "lr": 0.1, **(training or {})},
        data={"source": "memory", "x_var": "X", "y_var": "y", **(data or {})},
    )


def _ns(n=16):
    torch.manual_seed(0)
    return {"X": torch.randn(n, 8), "y": torch.randint(0, 3, (n,))}


# --- the sanitizer ----------------------------------------------------------

def test_finite_replaces_only_non_finite_floats():
    nan, inf = float("nan"), float("inf")
    assert _finite(nan) is None
    assert _finite(inf) is None
    assert _finite(-inf) is None
    assert _finite(0.0) == 0.0        # zero is finite, and must not be nulled
    assert _finite(-1.5) == -1.5
    assert _finite(3) == 3
    assert _finite("nan") == "nan"    # a string is not a float
    assert _finite(None) is None
    assert _finite(True) is True


def test_finite_recurses_through_the_message_shape():
    """Metrics arrive nested — {"metrics": {...}, "health": [{...}]} — so a
    shallow pass would miss exactly the values that carry the divergence."""
    msg = {
        "type": "run_epoch",
        "epoch": 7,
        "metrics": {"train_loss": float("nan"), "val_loss": 0.5},
        "health": [{"layer": "l", "dw": float("inf")}],
        "secs": 1.25,
    }
    out = _finite(msg)
    assert out["metrics"] == {"train_loss": None, "val_loss": 0.5}
    assert out["health"] == [{"layer": "l", "dw": None}]
    assert out["epoch"] == 7 and out["secs"] == 1.25
    # The whole thing must now survive a STRICT encoder — this is the property
    # that matters, since both transports refuse a bare NaN token.
    json.dumps(out, allow_nan=False)


# --- the transports ---------------------------------------------------------

def test_status_survives_a_nan_history():
    """GET /api/run/status uses JSONResponse, which refuses NaN and 500s — so
    the tab joining mid-divergence lost everything, including the finite epochs
    that show HOW it diverged."""
    mgr = RunManager()
    mgr.history = {"train_loss": [0.9, 0.4, float("nan")], "val_loss": [1.0, 0.5, float("inf")]}
    payload = json.dumps(mgr.status(), allow_nan=False)  # must not raise
    assert '"train_loss": [0.9, 0.4, null]' in payload
    assert '"val_loss": [1.0, 0.5, null]' in payload


def _nan_ns():
    """One NaN in the features — the everyday way a real run goes non-finite
    (an unclean column, a divide-by-zero upstream), and one pre-flight can't
    catch because the shapes and dtypes are all correct."""
    ns = _ns()
    ns["X"][0, 0] = float("nan")
    return ns


def test_emitted_frames_are_strict_json_even_when_the_loss_diverges():
    """The socket path: a run whose loss goes non-finite must still deliver
    every epoch. Previously the frame carrying the NaN was unparseable, so the
    stream died and the run went on to report success."""
    mgr = RunManager()
    events: list[dict] = []
    err = mgr.start(
        _mlp_graph({"epochs": 5}), namespace=_nan_ns(), emit=events.append
    )
    assert err is None and mgr.join(JOIN_TIMEOUT)

    epochs = [e for e in events if e["type"] == "run_epoch"]
    assert len(epochs) == 5, "every epoch must be emitted, divergence or not"
    for event in events:
        json.dumps(event, allow_nan=False)  # the actual regression

    # And the divergence is visible rather than silently dropped: with an lr
    # that large the loss must have left the finite range at some point.
    losses = [e["metrics"].get("train_loss") for e in epochs]
    assert any(v is None for v in losses), f"expected a diverged epoch, got {losses}"


def test_history_keeps_the_raw_value_for_the_kernel():
    """Sanitizing is a transport concern. `sess.history` in the notebook is
    plain Python and should still show the NaN — that IS the diagnosis."""
    mgr = RunManager()
    err = mgr.start(
        _mlp_graph({"epochs": 3}), namespace=_nan_ns(), emit=lambda m: None
    )
    assert err is None and mgr.join(JOIN_TIMEOUT)
    assert any(math.isnan(v) or math.isinf(v) for v in mgr.history["train_loss"])


# --- the traceback ----------------------------------------------------------

def test_a_failed_run_keeps_its_traceback():
    ns = _ns()
    ns["y"] = torch.randn(16, 2)  # float targets → CrossEntropyLoss blows up
    mgr = RunManager()
    err = mgr.start(_mlp_graph(), namespace=ns, emit=lambda m: None)
    assert err is None  # pre-flight can't know — it's a real tensor
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "failed"

    tb = mgr.error_traceback
    assert tb, "the frames are the whole point of catching here"
    assert "Traceback (most recent call last)" in tb
    # It must reach into the GENERATED source, not stop at the runner — that is
    # what linecache registration buys and what a bug report needs.
    assert "train" in tb
    assert mgr.error.split(":")[0] in tb  # the same exception, with its frames


def test_status_ships_the_traceback_and_a_new_run_clears_it():
    ns = _ns()
    ns["y"] = torch.randn(16, 2)
    mgr = RunManager()
    mgr.start(_mlp_graph(), namespace=ns, emit=lambda m: None)
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.status()["error_traceback"] == mgr.error_traceback

    # A clean run must not inherit the previous failure's frames.
    mgr.start(_mlp_graph(), namespace=_ns(), emit=lambda m: None)
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error
    assert mgr.status()["error_traceback"] is None
