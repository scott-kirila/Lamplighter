"""Training codegen: config → a clean train() function, and the generated loop
actually trains a model — including that validation is genuinely held out."""
import contextlib
import io
import re

import pytest
import torch
import torch.nn as nn

from backend.codegen import generate_module, generate_training
from backend.registry import available_devices
from backend.schema import Graph
from tests.helpers import edge, graph, node


def _code(training=None):
    return generate_training(Graph(training=training or {}))


def test_default_loss_and_optimizer():
    code = _code()
    assert "nn.CrossEntropyLoss()" in code
    assert "torch.optim.Adam(model.parameters(), lr=0.001)" in code
    assert "weight_decay" not in code  # default 0.0 omitted


def test_custom_loss_optimizer_and_weight_decay():
    code = _code({"loss": "MSELoss", "optimizer": "SGD", "lr": 0.05, "weight_decay": 1e-4})
    assert "nn.MSELoss()" in code
    assert "torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0001)" in code


def test_epochs_and_batch_size_baked_in():
    code = _code({"epochs": 5, "batch_size": 16})
    assert "def train(model, X, y, *, epochs=5, batch_size=16, device='auto'):" in code


def test_accuracy_for_classification():
    # Default metric=accuracy + a classification loss → top-1 accuracy reported.
    code = _code()  # default loss CrossEntropyLoss
    assert "out.argmax(dim=-1) == yb" in code
    assert "acc {train_acc:.3f}" in code


def test_accuracy_omitted_for_regression():
    # A regression loss never emits argmax accuracy, even with metric=accuracy.
    code = _code({"loss": "MSELoss", "metric": "accuracy"})
    assert "argmax" not in code
    assert "acc {" not in code


def test_metric_none_disables_accuracy():
    code = _code({"loss": "CrossEntropyLoss", "metric": "none"})
    assert "argmax" not in code


def test_val_split_adds_validation():
    code = _code({"val_split": 0.2})
    assert "def train(model, X, y, *, epochs=10, batch_size=32, val_split=0.2, device='auto'):" in code
    assert "X_val, y_val = X[val_idx], y[val_idx]" in code
    assert "val_loss = loss_fn(val_out, yv).item()" in code


def test_no_val_split_keeps_simple_signature():
    code = _code()  # val_split defaults to 0.0
    assert "val_split" not in code
    assert "X_train, y_train = X, y" in code


def test_generated_train_with_val_and_accuracy_runs():
    # exec a classifier train() with validation + accuracy; assert it runs and
    # the (printed) loop completes, returning the model.
    ns: dict = {}
    # Pin CPU so the assertion is host-independent (auto would pick the local
    # accelerator and leave the model there, breaking the post-hoc CPU forward).
    exec(_code({"epochs": 20, "batch_size": 8, "lr": 0.05, "val_split": 0.25, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X = torch.randn(40, 4)
    y = torch.randint(0, 3, (40,))

    before = nn.functional.cross_entropy(model(X), y).item()
    returned = train(model, X, y)
    after = nn.functional.cross_entropy(model.eval()(X), y).item()
    assert returned is model
    assert after < before


def test_generated_train_actually_trains():
    # exec the generated train(), run it on a tiny model, assert the loss drops.
    ns: dict = {}
    exec(_code({"epochs": 40, "batch_size": 8, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))

    before = nn.functional.cross_entropy(model(X), y).item()
    returned = train(model, X, y)
    after = nn.functional.cross_entropy(model(X), y).item()

    assert returned is model
    assert after < before


# --- Validation-integrity regression tests --------------------------------
# These exercise the *generated* loop end-to-end and guard against the worst
# failure mode: a validation set that secretly leaks training data.


def _build(classes, hidden, training):
    """Build the model + train() from a small MLP graph and a training config."""
    g = graph(
        [
            node("in", "Input", {"shape": "32, 64"}),
            node("a", "Linear", {"out_features": hidden}),
            node("r", "ReLU"),
            node("b", "Linear", {"out_features": classes}),
            node("out", "Output"),
        ],
        [edge("in", "a"), edge("a", "r"), edge("r", "b"), edge("b", "out")],
    )
    # Pin CPU so the numeric assertions are deterministic across hosts (auto would
    # run on the local accelerator, where float results can differ slightly).
    g.training = {"device": "cpu", **training}
    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    tns: dict = {}
    exec(generate_training(g), tns)  # noqa: S102
    return mns["GeneratedModel"](), tns["train"]


def _run_last_epoch(train, model, X, y):
    """Run training and parse the final epoch's printed metrics."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(model, X, y)
    line = buf.getvalue().splitlines()[-1]
    return {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", line)}


def test_validation_does_not_leak():
    # Random labels (independent of features): the model can memorize the
    # TRAINING set, but a genuinely held-out val set cannot beat chance (1/10).
    # If the split leaked, val_acc would climb with train_acc.
    model, train = _build(classes=10, hidden=64, training={
        "epochs": 100, "batch_size": 16, "lr": 0.01, "val_split": 0.25, "metric": "accuracy"})
    torch.manual_seed(0)
    X = torch.randn(100, 64)
    y = torch.randint(0, 10, (100,))
    m = _run_last_epoch(train, model, X, y)
    assert m["acc"] >= 0.5        # train memorized the random labels
    assert m["val_acc"] <= 0.35   # val stayed near chance (0.10) — no leakage


def test_validation_reflects_generalization():
    # Learnable data + 20% label noise → a real ~80% ceiling. Val must plateau
    # below 100% (a trivial/leaking val would not).
    model, train = _build(classes=10, hidden=32, training={
        "epochs": 25, "batch_size": 32, "lr": 0.001, "val_split": 0.2, "metric": "accuracy"})
    torch.manual_seed(0)
    centers = torch.randn(10, 64)
    y = torch.randint(0, 10, (1000,))
    X = centers[y] + torch.randn(1000, 64)
    flip = torch.rand(1000) < 0.20
    y[flip] = torch.randint(0, 10, (int(flip.sum()),))
    m = _run_last_epoch(train, model, X, y)
    assert 0.6 < m["val_acc"] < 0.92  # real ceiling — neither chance nor 100%


def test_val_split_is_a_disjoint_partition():
    # The split must partition the data: perm[:split] / perm[split:] never overlap.
    code = generate_training(Graph(training={"val_split": 0.2}))
    assert "train_idx, val_idx = perm[:split], perm[split:]" in code


# --- device selection (Training v2) ---------------------------------------

def test_device_defaults_to_auto_and_resolves():
    code = _code()  # default device
    assert "device='auto'" in code
    # "auto" resolution prefers CUDA, then a guarded MPS, else CPU.
    assert "if torch.cuda.is_available():" in code
    assert 'getattr(torch.backends, "mps", None) is not None' in code
    assert "model = model.to(device)" in code
    # Batches move to the device each step.
    assert "xb, yb = X_train[idx].to(device), y_train[idx].to(device)" in code


def test_specific_device_is_baked_as_default():
    code = _code({"device": "cuda"})
    assert "def train(model, X, y, *, epochs=10, batch_size=32, device='cuda'):" in code
    # The resolver still runs, so a specific choice is wrapped in torch.device.
    assert "device = torch.device(device)" in code


def test_generated_train_runs_on_cpu_device():
    # End-to-end: an explicit device='cpu' train() still trains (loss drops).
    ns: dict = {}
    exec(_code({"epochs": 40, "batch_size": 8, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X, y = torch.randn(16, 4), torch.randint(0, 3, (16,))
    before = nn.functional.cross_entropy(model(X), y).item()
    train(model, X, y)
    assert nn.functional.cross_entropy(model(X), y).item() < before


# --- device detection, without hardware (monkeypatched torch flags) --------
# available_devices() drives what the training form offers; these pin its branch
# logic so a CPU-only CI run still covers the cuda/mps detection paths.


def _fake_mps(available: bool):
    """A stand-in torch.backends.mps whose is_available() is fixed."""
    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return available
    return _MPS


def test_available_devices_lists_all_when_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert available_devices() == ["auto", "cpu", "cuda", "mps"]


def test_available_devices_cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(False), raising=False)
    assert available_devices() == ["auto", "cpu"]


def test_available_devices_handles_missing_mps_namespace(monkeypatch):
    # An older torch exposes no usable mps backend — getattr(..., None) yields
    # None and the guard must tolerate it (the version concern this targets).
    # (torch lazily re-imports torch.backends.mps, so we set None rather than
    # delattr — None is exactly what the getattr guard sees when it's absent.)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    assert available_devices() == ["auto", "cpu"]


def test_auto_resolver_falls_through_to_cpu(monkeypatch):
    # Run the *generated* auto-resolver with no accelerator available and confirm
    # the model actually lands on CPU — exercises the resolver end-to-end.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    ns: dict = {}
    exec(_code({"epochs": 1, "device": "auto"}), ns)  # noqa: S102
    model = nn.Linear(4, 3)
    ns["train"](model, torch.randn(8, 4), torch.randint(0, 3, (8,)))
    assert next(model.parameters()).device.type == "cpu"


# --- real device runtime (auto-skips when the hardware isn't present) -------
# The same test covers whatever the host has: CPU everywhere, MPS on a Mac,
# CUDA on a GPU box. Catches device-specific failures (e.g. an MPS op gap) that
# string/codegen tests can't.


def _host_has(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    return False


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_train_runs_on_real_device(device):
    if not _host_has(device):
        pytest.skip(f"{device} not available on this host")
    ns: dict = {}
    exec(_code({"epochs": 5, "batch_size": 8, "lr": 0.05, "device": device}), ns)  # noqa: S102
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X, y = torch.randn(16, 4), torch.randint(0, 3, (16,))
    ns["train"](model, X, y)  # a real forward/backward on the device
    assert next(model.parameters()).device.type == device
