"""Training codegen: config → a clean train() function, and the generated loop
actually trains a model — including that validation is genuinely held out."""
import contextlib
import io
import re

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lamplighter.backend.codegen import generate_module, generate_training
from lamplighter.backend.registry import available_devices
from lamplighter.backend.schema import Graph
from tests.helpers import edge, graph, node, single_model_project


def _code(training=None, data=None):
    # training is project-level; data (batch_size / val_split) lives on the data
    # node and never reaches train() — passed here only to document that.
    return generate_training(Graph(), training or {})


def test_default_loss_and_optimizer():
    code = _code()
    assert "nn.CrossEntropyLoss()" in code
    assert "torch.optim.Adam(model.parameters(), lr=0.001)" in code
    assert "weight_decay" not in code  # default 0.0 omitted


def test_custom_loss_optimizer_and_weight_decay():
    code = _code({"loss": "MSELoss", "optimizer": "SGD", "lr": 0.05, "weight_decay": 1e-4})
    assert "nn.MSELoss()" in code
    # SGD carries the working momentum default — momentumless SGD is a trap.
    assert "torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)" in code


def test_momentum_only_for_the_optimizers_that_take_it():
    # Adam's momentum lives in betas — the knob must never leak into its call,
    # whatever the (defaulted) config says.
    assert "momentum" not in _code({"optimizer": "Adam"})
    assert "momentum" not in _code({"optimizer": "AdamW", "momentum": 0.9})
    # SGD/RMSprop take it; an explicit 0 emits nothing (torch's own default).
    assert "torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)" in _code({"optimizer": "RMSprop"})
    assert "momentum" not in _code({"optimizer": "SGD", "momentum": 0.0})
    assert "torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)" in _code(
        {"optimizer": "SGD", "momentum": 0.5}
    )


def test_signature_is_always_the_loader_form():
    # One data path: train(model, loader) with an optional val_loader. Batching
    # and the val split live in make_dataloaders (the Data panel), never here.
    code = _code({"epochs": 5})
    assert "def train(model, loader, *, epochs=5, val_loader=None, device='auto', on_epoch=None, on_step=None):" in code
    assert "batch_size" not in code and "val_split" not in code


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


def _make(data=None):
    """exec the generated make_dataloaders() for a data config — the same source
    the Data panel shows, so these tests exercise the real pipeline."""
    from lamplighter.backend.codegen import generate_dataloader

    ns: dict = {}
    exec(generate_dataloader(Graph(), data or {}), ns)  # noqa: S102
    return ns["make_dataloaders"]


def test_generated_train_with_val_and_accuracy_runs():
    # The full generated pipeline: make_dataloaders (with a val split) feeding
    # train(); history carries all four series, and the model actually learns.
    ns: dict = {}
    # Pin CPU so the assertion is host-independent (auto would pick the local
    # accelerator and leave the model there, breaking the post-hoc CPU forward).
    exec(_code({"epochs": 20, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X = torch.randn(40, 4)
    y = torch.randint(0, 3, (40,))
    train_loader, val_loader = _make({"batch_size": 8, "val_split": 0.25})(X, y)

    before = nn.functional.cross_entropy(model(X), y).item()
    history = train(model, train_loader, val_loader=val_loader)
    after = nn.functional.cross_entropy(model.eval()(X), y).item()
    assert after < before
    # train() returns a per-epoch history (model is trained in place).
    assert set(history) == {"train_loss", "train_acc", "val_loss", "val_acc"}
    assert all(len(v) == 20 for v in history.values())


def test_generated_train_actually_trains():
    # exec the generated train(), run it on a tiny model, assert the loss drops.
    ns: dict = {}
    exec(_code({"epochs": 40, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    loader, _ = _make({"batch_size": 8})(X, y)

    before = nn.functional.cross_entropy(model(X), y).item()
    history = train(model, loader)
    after = nn.functional.cross_entropy(model(X), y).item()

    assert after < before
    assert len(history["train_loss"]) == 40  # one entry per epoch


def test_generated_train_prints_only_when_standalone(capsys):
    # The in-app runner drives train() with an on_epoch hook and reports progress
    # itself, so the loop's print must stay quiet there — otherwise every epoch
    # leaks into the notebook that started the session. Standalone (no hook), it
    # still prints per-epoch progress.
    ns: dict = {}
    exec(_code({"epochs": 2, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))

    # Driven by a hook (what the app does): silent.
    train(nn.Linear(4, 3), _make({"batch_size": 8})(X, y)[0], on_epoch=lambda *_: True)
    assert capsys.readouterr().out == ""

    # Standalone: one printed line per epoch.
    train(nn.Linear(4, 3), _make({"batch_size": 8})(X, y)[0])
    assert capsys.readouterr().out.count("epoch") == 2


# --- Validation-integrity regression tests --------------------------------
# These exercise the *generated* loop end-to-end and guard against the worst
# failure mode: a validation set that secretly leaks training data.


def _build(classes, hidden, training, data=None):
    """Build the model + train() + make_dataloaders() from a small MLP graph and
    training/data config — the exact generated pipeline the Run button executes."""
    from lamplighter.backend.codegen import generate_dataloader

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
    training = {"device": "cpu", **training}
    data = data or {}
    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    tns: dict = {}
    exec(generate_training(g, training), tns)  # noqa: S102
    dns: dict = {}
    exec(generate_dataloader(g, data), dns)  # noqa: S102
    return mns["GeneratedModel"](), tns["train"], dns["make_dataloaders"]


def _run_last_epoch(train, make, model, X, y):
    """Run the generated pipeline and parse the final epoch's printed metrics."""
    train_loader, val_loader = make(X, y)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(model, train_loader, val_loader=val_loader)
    line = buf.getvalue().splitlines()[-1]
    return {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", line)}


def test_validation_does_not_leak():
    # Random labels (independent of features): the model can memorize the
    # TRAINING set, but a genuinely held-out val set cannot beat chance (1/10).
    # If the split leaked, val_acc would climb with train_acc.
    model, train, make = _build(classes=10, hidden=64, training={
        "epochs": 100, "lr": 0.01, "metric": "accuracy"},
        data={"batch_size": 16, "val_split": 0.25})
    torch.manual_seed(0)
    X = torch.randn(100, 64)
    y = torch.randint(0, 10, (100,))
    m = _run_last_epoch(train, make, model, X, y)
    assert m["acc"] >= 0.5        # train memorized the random labels
    assert m["val_acc"] <= 0.35   # val stayed near chance (0.10) — no leakage


def test_validation_reflects_generalization():
    # Learnable data + 20% label noise → a real ~80% ceiling. Val must plateau
    # below 100% (a trivial/leaking val would not).
    model, train, make = _build(classes=10, hidden=32, training={
        "epochs": 25, "lr": 0.001, "metric": "accuracy"},
        data={"batch_size": 32, "val_split": 0.2})
    torch.manual_seed(0)
    centers = torch.randn(10, 64)
    y = torch.randint(0, 10, (1000,))
    X = centers[y] + torch.randn(1000, 64)
    flip = torch.rand(1000) < 0.20
    y[flip] = torch.randint(0, 10, (int(flip.sum()),))
    m = _run_last_epoch(train, make, model, X, y)
    assert 0.6 < m["val_acc"] < 0.92  # real ceiling — neither chance nor 100%


def test_post_training_code_reflects_posted_graph():
    # The Training code panel POSTs the live graph so the preview matches the
    # canvas without state-sync timing.
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    project = single_model_project(Graph(), training={"epochs": 7})
    with TestClient(app) as c:
        code = c.post("/api/training/code", json=project.model_dump()).json()["code"]
    assert "epochs=7" in code


def test_val_split_and_batch_size_live_only_in_make_dataloaders():
    # One data path: batching and the val split are make_dataloaders' business;
    # train() never mentions them, no matter where the values are set.
    from lamplighter.backend.codegen import generate_dataloader

    trainer = generate_training(Graph(), {})
    assert "val_split" not in trainer and "batch_size" not in trainer
    loaders = generate_dataloader(Graph(), {"val_split": 0.25, "batch_size": 16})
    assert "val_split=0.25" in loaders and "batch_size=16" in loaders
    stale = generate_training(Graph(), {"val_split": 0.25, "batch_size": 16})
    assert "val_split" not in stale and "batch_size" not in stale  # dead location


# --- on_epoch hook (run-from-app) ------------------------------------------
# One optional param serves progress reporting, cooperative stop (return False),
# and user-side early stopping. Only an explicit False stops (None continues).


def _tiny_loader(n=12, feats=4, classes=3):
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, feats), torch.randint(0, classes, (n,))), batch_size=8)


def test_on_epoch_in_signature():
    # Both progress hooks live in the signature (on_epoch, and the per-step on_step).
    assert "on_epoch=None, on_step=None):" in _code()


def test_on_epoch_called_per_epoch_after_history_appends():
    ns: dict = {}
    exec(_code({"epochs": 3, "device": "cpu"}), ns)  # noqa: S102
    calls: list = []
    # append returns None (not False) — training must run to completion.
    ns["train"](nn.Linear(4, 3), _tiny_loader(), on_epoch=lambda e, h: calls.append((e, len(h["train_loss"]))))
    assert calls == [(1, 1), (2, 2), (3, 3)]  # fires per epoch, after the appends


def test_on_epoch_false_stops_early():
    ns: dict = {}
    exec(_code({"epochs": 50, "device": "cpu"}), ns)  # noqa: S102
    history = ns["train"](nn.Linear(4, 3), _tiny_loader(), on_epoch=lambda e, h: e < 5)  # False at 5
    assert len(history["train_loss"]) == 5  # partial history from the early stop


# --- returned history (Training v2) ---------------------------------------

def test_history_regression_has_loss_only():
    # No accuracy for a regression loss -> history carries just the loss series.
    ns: dict = {}
    exec(_code({"loss": "MSELoss", "epochs": 3, "device": "cpu"}), ns)  # noqa: S102
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(12, 4), torch.randn(12, 1)), batch_size=8)
    history = ns["train"](nn.Linear(4, 1), loader)
    assert set(history) == {"train_loss", "val_loss"}  # val key present, unused
    assert history["val_loss"] == []
    assert len(history["train_loss"]) == 3 and all(isinstance(v, float) for v in history["train_loss"])


def test_history_val_only_on_val_epochs():
    ns: dict = {}
    exec(_code({"epochs": 2, "device": "cpu"}), ns)  # noqa: S102
    model = nn.Linear(4, 3)
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))
    loader = DataLoader(ds, batch_size=8)
    # Without a val_loader, val keys exist but stay empty; train series fills.
    h1 = ns["train"](model, loader)
    assert len(h1["train_loss"]) == 2 and h1["val_loss"] == []
    # With a val_loader, val series fills too.
    h2 = ns["train"](model, loader, val_loader=loader)
    assert len(h2["val_loss"]) == 2 and len(h2["val_acc"]) == 2


# --- device selection (Training v2) ---------------------------------------

def test_device_defaults_to_auto_and_resolves():
    code = _code()  # default device
    assert "device='auto'" in code
    # "auto" resolution prefers CUDA, then a guarded MPS, else CPU.
    assert "if torch.cuda.is_available():" in code
    assert 'getattr(torch.backends, "mps", None) is not None' in code
    assert "model = model.to(device)" in code
    # Batches move to the device each step.
    assert "xb = xb.to(device)" in code and "yb = yb.to(device)" in code


def test_specific_device_is_baked_as_default():
    code = _code({"device": "cuda"})
    assert "def train(model, loader, *, epochs=10, val_loader=None, device='cuda', on_epoch=None, on_step=None):" in code
    # The resolver still runs, so a specific choice is wrapped in torch.device.
    assert "device = torch.device(device)" in code


def test_generated_train_runs_on_cpu_device():
    # End-to-end: an explicit device='cpu' train() still trains (loss drops).
    ns: dict = {}
    exec(_code({"epochs": 40, "lr": 0.05, "device": "cpu"}), ns)  # noqa: S102
    train = ns["train"]
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X, y = torch.randn(16, 4), torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    before = nn.functional.cross_entropy(model(X), y).item()
    train(model, loader)
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
    ns["train"](model, _tiny_loader(n=8))
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
    exec(_code({"epochs": 5, "lr": 0.05, "device": device}), ns)  # noqa: S102
    model = nn.Linear(4, 3)
    ns["train"](model, _tiny_loader(n=16))  # a real forward/backward on the device
    assert next(model.parameters()).device.type == device


# --- the loader loop --------------------------------------------------------

def test_loader_loop_shape():
    code = _code()
    assert "for batch in loader:" in code
    assert "xb, yb = batch" in code
    assert "batch_size" not in code  # the loader owns batching


def test_single_input_trains():
    model, train, _ = _build(classes=3, hidden=16, training={"epochs": 30, "lr": 0.05})
    torch.manual_seed(0)
    X, y = torch.randn(40, 64), torch.randint(0, 3, (40,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    before = nn.functional.cross_entropy(model(X), y).item()
    train(model, loader)
    assert nn.functional.cross_entropy(model(X), y).item() < before


def test_val_loader_reports_val_metrics():
    model, train, _ = _build(classes=3, hidden=16, training={
        "epochs": 2, "lr": 0.05, "metric": "accuracy"})
    torch.manual_seed(0)
    X, y = torch.randn(40, 64), torch.randint(0, 3, (40,))
    tl = DataLoader(TensorDataset(X[:30], y[:30]), batch_size=8)
    vl = DataLoader(TensorDataset(X[30:], y[30:]), batch_size=8)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(model, tl, val_loader=vl)
    out = buf.getvalue()
    assert "val_loss" in out and "val_acc" in out


def test_omits_validation_without_val_loader():
    model, train, _ = _build(classes=3, hidden=16, training={"epochs": 1})
    torch.manual_seed(0)
    X, y = torch.randn(24, 64), torch.randint(0, 3, (24,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(model, loader)  # no val_loader
    assert "val_loss" not in buf.getvalue()


def test_multi_input_loader_unpacking():
    # A two-input model + a DataLoader yielding (x0, x1, y): `*xb, yb = batch`
    # unpacks the trailing target, the rest feed model(*xb).
    g = graph(
        [
            node("a", "Input", {"shape": "8, 8"}, y=0),
            node("b", "Input", {"shape": "8, 8"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"),
         edge("cat", "lin"), edge("lin", "out")],
    )
    code = generate_training(g, {"epochs": 2, "device": "cpu"})
    assert "*xb, yb = batch" in code
    assert "out = model(*xb)" in code

    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    tns: dict = {}
    exec(code, tns)  # noqa: S102
    X0, X1, y = torch.randn(24, 8), torch.randn(24, 8), torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(X0, X1, y), batch_size=8)
    tns["train"](mns["GeneratedModel"](), loader)  # runs a real forward/backward


# --- LR schedulers ---------------------------------------------------------------

def _lr_history(training, X=None, y=None, val_split=0.0):
    """exec the generated train() and return its history (CPU, tiny model)."""
    ns: dict = {}
    exec(_code({"device": "cpu", **training}), ns)  # noqa: S102
    torch.manual_seed(0)
    X = torch.randn(24, 4) if X is None else X
    y = torch.randint(0, 3, (24,)) if y is None else y
    train_loader, val_loader = _make({"batch_size": 8, "val_split": val_split})(X, y)
    return ns["train"](nn.Linear(4, 3), train_loader, val_loader=val_loader)


def test_no_scheduler_emits_nothing():
    src = _code({"epochs": 3})
    assert "sched" not in src and '"lr"' not in src  # the default is untouched


def test_steplr_decays_on_schedule():
    src = _code({"scheduler": "StepLR", "step_size": 1, "gamma": 0.5, "lr": 1e-3, "epochs": 3})
    assert "torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)" in src

    history = _lr_history({"scheduler": "StepLR", "step_size": 1, "gamma": 0.5, "lr": 1e-3, "epochs": 3})
    # The lr each epoch trained at: initial, then halved per epoch.
    assert history["lr"] == [1e-3, 5e-4, 2.5e-4]


def test_cosine_anneals_over_the_runs_epochs():
    src = _code({"scheduler": "CosineAnnealingLR", "epochs": 5})
    assert "CosineAnnealingLR(opt, T_max=epochs)" in src  # spans exactly this run

    history = _lr_history({"scheduler": "CosineAnnealingLR", "lr": 1e-2, "epochs": 5})
    lrs = history["lr"]
    assert lrs[0] == 1e-2  # starts at the configured lr
    assert all(a > b for a, b in zip(lrs, lrs[1:]))  # strictly anneals downward


def test_plateau_steps_on_val_loss_with_train_fallback():
    src = _code({"scheduler": "ReduceLROnPlateau", "plateau_factor": 0.5, "plateau_patience": 0})
    assert "ReduceLROnPlateau(opt, factor=0.5, patience=0)" in src
    assert 'sched.step(history["val_loss"][-1] if val_loader is not None else train_loss)' in src

    cfg = {"scheduler": "ReduceLROnPlateau", "plateau_factor": 0.5,
           "plateau_patience": 0, "lr": 1e-3, "epochs": 3}
    with_val = _lr_history(cfg, val_split=0.25)
    without_val = _lr_history(cfg)
    # Both paths run and record the lr series (3 epochs each).
    assert len(with_val["lr"]) == 3 and len(without_val["lr"]) == 3


def test_onecycle_steps_per_batch_with_the_form_lr_as_peak():
    src = _code({"scheduler": "OneCycleLR", "lr": 1e-2, "epochs": 4})
    # Sized to the whole run from the loader, peak = the form lr.
    assert "OneCycleLR(opt, max_lr=0.01, epochs=epochs, steps_per_epoch=len(loader))" in src
    # Steps per BATCH — inside the loop, right after the optimizer step — and
    # the epoch tail records the lr without stepping again.
    assert "            opt.step()\n            sched.step()" in src
    assert src.count("sched.step()") == 1

    history = _lr_history({"scheduler": "OneCycleLR", "lr": 1e-2, "epochs": 4})
    lrs = history["lr"]
    assert len(lrs) == 4
    assert max(lrs) <= 1e-2 + 1e-9  # never exceeds the configured peak
    assert lrs[-1] < lrs[0]  # annealed well below where it started


def test_unknown_scheduler_is_rejected():
    # The name lands in the source as an attribute, so it's validated (the
    # torchvision-dataset rule) — a raw API caller can't inject through it.
    with pytest.raises(ValueError, match="unknown LR scheduler"):
        _code({"scheduler": "StepLR); import os #"})


# --- grad clipping + mixed precision ------------------------------------------------

def test_defaults_emit_no_clip_or_amp():
    src = _code({"epochs": 3})
    assert "clip_grad_norm_" not in src and "GradScaler" not in src and "autocast" not in src


def test_clipping_lands_between_backward_and_step():
    src = _code({"clip_grad_norm": 0.5})
    clip = "torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)"
    assert src.index("loss.backward()") < src.index(clip) < src.index("opt.step()")

    # And it genuinely constrains the update: with SGD (whose step is lr·grad —
    # Adam's moment normalization would mask the clip) a near-zero max_norm
    # freezes the model relative to an unclipped run from the same seed.
    def delta(training):
        ns: dict = {}
        exec(_code({"device": "cpu", "optimizer": "SGD", "lr": 0.1, "epochs": 1, **training}), ns)  # noqa: S102
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        torch.manual_seed(1)
        X, y = torch.randn(24, 4), torch.randint(0, 3, (24,))
        loader, _ = _make({"batch_size": 8})(X, y)
        ns["train"](model, loader)
        return sum((model.state_dict()[k] - before[k]).abs().sum().item() for k in before)

    assert delta({"clip_grad_norm": 1e-6}) < delta({}) * 0.01


def test_amp_generates_the_scaled_loop_and_runs_on_cpu():
    src = _code({"amp": True, "device": "cpu"})
    assert "scaler = torch.amp.GradScaler(device.type)" in src
    assert "with torch.autocast(device_type=device.type):" in src
    assert "scaler.scale(loss).backward()" in src
    assert src.index("scaler.step(opt)") < src.index("scaler.update()")
    assert "loss.backward()" not in src and "opt.step()" not in src  # fully replaced

    ns: dict = {}
    exec(src, ns)  # noqa: S102
    torch.manual_seed(0)
    X, y = torch.randn(24, 4), torch.randint(0, 3, (24,))
    train_loader, val_loader = _make({"batch_size": 8, "val_split": 0.25})(X, y)
    history = ns["train"](nn.Linear(4, 3), train_loader, val_loader=val_loader, epochs=2)
    assert len(history["train_loss"]) == 2
    assert all(v == v for v in history["train_loss"] + history["val_loss"])  # finite


def test_amp_clipping_unscales_first():
    src = _code({"amp": True, "clip_grad_norm": 1.0})
    clip = "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"
    # The AMP-correct order: scaled backward → unscale → clip → scaler step.
    assert (
        src.index("scaler.scale(loss).backward()")
        < src.index("scaler.unscale_(opt)")
        < src.index(clip)
        < src.index("scaler.step(opt)")
    )


# --- metrics beyond accuracy --------------------------------------------------------

def _frozen_run(training, X, y, val_split=0.0):
    """One epoch at lr=0 (SGD: the model never moves), so the reported metric is
    computable by hand from the seeded model's raw outputs."""
    ns: dict = {}
    exec(_code({"device": "cpu", "optimizer": "SGD", "lr": 0.0, "epochs": 1, **training}), ns)  # noqa: S102
    torch.manual_seed(0)
    model = nn.Linear(4, 6)
    loaders = _make({"batch_size": 8, "val_split": val_split})(X, y)
    history = ns["train"](model, loaders[0], val_loader=loaders[1])
    return history, model


def test_top5_accuracy_matches_a_hand_computation():
    torch.manual_seed(1)
    X, y = torch.randn(24, 4), torch.randint(0, 6, (24,))
    history, model = _frozen_run({"metric": "top5_accuracy"}, X, y)
    with torch.no_grad():
        top5 = model(X).topk(5, dim=-1).indices
    expected = (top5 == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    assert history["train_top5"] == [pytest.approx(expected)]


def test_macro_f1_matches_a_hand_computation():
    torch.manual_seed(1)
    X, y = torch.randn(24, 4), torch.randint(0, 6, (24,))
    history, model = _frozen_run({"metric": "macro_f1"}, X, y)
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)
    # Independent per-class F1, the long way.
    scores = []
    for c in range(max(int(pred.max()), int(y.max())) + 1):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    assert history["train_f1"] == [pytest.approx(sum(scores) / len(scores))]


def test_mae_matches_a_hand_computation_and_rides_val():
    torch.manual_seed(1)
    X, y = torch.randn(24, 4), torch.randn(24, 6)
    # No split: the train loader sees every sample, so the hand value is exact.
    history, model = _frozen_run({"metric": "mae", "loss": "MSELoss"}, X, y)
    with torch.no_grad():
        expected = (model(X) - y).abs().mean().item()
    assert history["train_mae"] == [pytest.approx(expected, rel=1e-5)]

    # With a split, the val variant rides along.
    history, _ = _frozen_run({"metric": "mae", "loss": "MSELoss"}, X, y, val_split=0.25)
    assert len(history["val_mae"]) == 1


def test_metrics_gate_on_their_losses():
    # Classification metrics never emit under a regression loss, and vice versa.
    assert "top5" not in _code({"metric": "top5_accuracy", "loss": "MSELoss"})
    assert "f1" not in _code({"metric": "macro_f1", "loss": "L1Loss"})
    assert "mae" not in _code({"metric": "mae", "loss": "CrossEntropyLoss"})
    # And the gated pick still trains — loss-only, like accuracy's precedent.
    src = _code({"metric": "mae", "loss": "CrossEntropyLoss"})
    assert 'history = {"train_loss": [], "val_loss": []}' in src


# --- the loss surface: curated additions + the Custom hatch -------------------

def test_huber_loss_emits_and_reports_mae():
    src = _code({"loss": "HuberLoss", "metric": "mae"})
    assert "loss_fn = nn.HuberLoss()" in src
    assert "train_mae" in src  # a regression loss, so MAE is meaningful


def test_label_smoothing_rides_cross_entropy_only_when_set():
    assert "label_smoothing" not in _code({"loss": "CrossEntropyLoss"})  # 0 = torch's default
    src = _code({"loss": "CrossEntropyLoss", "label_smoothing": 0.1})
    assert "loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)" in src
    # It's a CE-only knob — never leaks onto another loss.
    assert "label_smoothing" not in _code({"loss": "MSELoss", "label_smoothing": 0.1})


def test_unknown_loss_or_optimizer_is_rejected():
    # Both land in the source as attributes (nn.X / torch.optim.X), so they're
    # validated, not escaped — the scheduler/dataset-name rule.
    with pytest.raises(ValueError, match="unknown loss 'Evil'"):
        _code({"loss": "Evil"})
    with pytest.raises(ValueError, match="unknown optimizer 'Evil'"):
        _code({"optimizer": "Evil"})


class WeightedMSE(nn.Module):
    """A registered custom loss (module-level so inspect.getsource can read it)."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, out, target):
        return ((out - target) ** 2).mean() * self.scale


def test_custom_loss_splices_its_source_and_trains():
    from lamplighter.backend import datastore

    try:
        datastore.register_modules(WeightedMSE=WeightedMSE)
        src = _code({"loss": "Custom", "loss_cls": "WeightedMSE", "loss_args": "scale=2.0",
                     "device": "cpu", "epochs": 1, "metric": "none"})
        # Spliced verbatim ABOVE train() — the source runs standalone, so an
        # eject/checkpoint stays self-contained (the Custom node's rule).
        assert "class WeightedMSE(nn.Module):" in src
        assert src.index("class WeightedMSE") < src.index("def train(")
        assert "loss_fn = WeightedMSE(scale=2.0)" in src

        ns: dict = {}
        exec(src, ns)  # noqa: S102
        torch.manual_seed(0)
        X, y = torch.randn(24, 4), torch.randn(24, 3)
        loader, _ = _make({"batch_size": 8})(X, y)
        history = ns["train"](nn.Linear(4, 3), loader)
        assert len(history["train_loss"]) == 1 and history["train_loss"][0] > 0
    finally:
        datastore.clear_modules()


def test_custom_loss_without_a_pick_says_what_to_do():
    with pytest.raises(ValueError, match="pick a registered module"):
        _code({"loss": "Custom"})
    with pytest.raises(ValueError, match="is not registered"):
        _code({"loss": "Custom", "loss_cls": "Nope"})


# --- gradient accumulation ----------------------------------------------------

def test_accumulation_off_by_default_emits_the_plain_loop():
    src = _code({"epochs": 1})
    assert "micro" not in src and "for batch in loader:" in src


def test_accumulation_steps_at_boundaries_and_flushes_the_tail():
    src = _code({"accumulate_steps": 4})
    assert "(loss / 4).backward()" in src  # scaled so gradients AVERAGE
    assert "if micro % 4 == 0:" in src
    assert "if micro % 4:  # flush the ragged tail" in src
    # micro is initialized before the loop, so an empty loader flushes nothing
    # (rather than referencing an unbound name).
    assert src.index("micro = 0") < src.index("for batch in loader:")
    # The reported per-batch loss stays UNscaled — curves are comparable.
    assert "batch_loss = loss.item()" in src


def test_accumulation_matches_one_large_batch():
    # The semantic claim: N micro-batches of size B/N accumulate to the same
    # update as one batch of size B. Same seed, same data, same order.
    def trained(training, batch_size):
        ns: dict = {}
        exec(_code({"device": "cpu", "optimizer": "SGD", "lr": 0.1, "epochs": 1,
                    "loss": "MSELoss", "metric": "none", **training}), ns)  # noqa: S102
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        torch.manual_seed(1)
        X, y = torch.randn(24, 4), torch.randn(24, 3)
        loader, _ = _make({"batch_size": batch_size, "shuffle": False})(X, y)
        ns["train"](model, loader)
        return model.state_dict()

    big = trained({}, 24)  # one batch of 24
    accumulated = trained({"accumulate_steps": 4}, 6)  # 4 × 6, one step
    for k in big:
        assert torch.allclose(big[k], accumulated[k], atol=1e-6), k


def test_accumulation_composes_with_clipping_amp_and_onecycle():
    # Clip/step ops move INSIDE the boundary (and the flush), unscaling first
    # under AMP; OneCycle is sized in optimizer steps, not batches.
    src = _code({"accumulate_steps": 3, "clip_grad_norm": 1.0, "amp": True,
                 "scheduler": "OneCycleLR"})
    assert "steps_per_epoch=(len(loader) + 2) // 3" in src
    assert "scaler.scale((loss / 3)).backward()" in src
    boundary = src.index("if micro % 3 == 0:")
    flush = src.index("if micro % 3:  # flush the ragged tail")
    for op in ("scaler.unscale_(opt)", "clip_grad_norm_(model.parameters(), 1.0)",
               "scaler.step(opt)", "sched.step()"):
        assert boundary < src.index(op) < flush  # inside the boundary block
        assert src.index(op, flush) > flush  # and repeated in the flush


# --- class imbalance: weighting the loss ---------------------------------------

def _imbalanced(n0=90, n1=9, n2=1, feats=4):
    """A 3-class split skewed 90:9:1 — the shape imbalance advice is about."""
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1), torch.full((n2,), 2)]).long()
    return torch.randn(len(y), feats), y


def test_class_weights_off_by_default_emits_nothing():
    src = _code({"epochs": 1})
    assert "label_counts" not in src and "weight=weight" not in src


def test_class_weights_are_inverse_frequency_over_the_training_split():
    src = _code({"class_weights": True, "device": "cpu", "epochs": 1, "metric": "none"})
    assert "loss_fn = nn.CrossEntropyLoss(weight=weight)" in src
    # Sized to the MODEL's logits (not the observed labels), so a class missing
    # from the split can't shorten the vector into a mid-run crash.
    assert "n_classes = model(*[t[:1].to(device) for t in probe[:-1]]).size(-1)" in src

    ns: dict = {}
    exec(src, ns)  # noqa: S102
    X, y = _imbalanced()
    loader, _ = _make({"batch_size": 10})(X, y)
    counts = ns["label_counts"](loader, 3)
    assert counts.tolist() == [90.0, 9.0, 1.0]
    weight = counts.sum() / (3 * counts.clamp(min=1.0))
    assert weight.tolist() == pytest.approx([100 / 270, 100 / 27, 100 / 3])
    # And it trains (the weighted loss is wired into the real loop).
    assert len(ns["train"](nn.Linear(4, 3), loader)["train_loss"]) == 1


def test_label_counts_reads_the_split_not_the_whole_dataset():
    # The weights must describe what the model actually TRAINS on: a Subset
    # from random_split reports its own labels, via the tensors fast path.
    src = _code({"class_weights": True, "device": "cpu", "epochs": 1, "metric": "none"})
    ns: dict = {}
    exec(src, ns)  # noqa: S102
    X, y = _imbalanced()
    train_loader, val_loader = _make({"batch_size": 10, "val_split": 0.25})(X, y)
    counts = ns["label_counts"](train_loader, 3)
    assert counts.sum().item() == 75  # the train split, not all 100
    assert counts.sum().item() == len(train_loader.dataset)


def test_label_counts_falls_back_to_a_batch_pass_for_an_opaque_dataset():
    # No .targets, no .tensors (a user's own Dataset) — one honest pass.
    class Opaque(torch.utils.data.Dataset):
        def __init__(self, y):
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return torch.randn(4), self.y[i]

    ns: dict = {}
    exec(_code({"class_weights": True, "device": "cpu"}), ns)  # noqa: S102
    _, y = _imbalanced()
    loader = DataLoader(Opaque(y), batch_size=8)
    assert ns["label_counts"](loader, 3).tolist() == [90.0, 9.0, 1.0]


def test_bce_class_weights_use_pos_weight_not_a_vector():
    # BCEWithLogits takes a positive-class SCALE, not a per-class vector —
    # different argument, different arithmetic.
    src = _code({"loss": "BCEWithLogitsLoss", "class_weights": True, "device": "cpu",
                 "epochs": 1, "metric": "none"})
    assert "loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)" in src
    assert "weight=weight" not in src
    assert "probe" not in src  # no logit probe needed: the classes are 0/1

    ns: dict = {}
    exec(src, ns)  # noqa: S102
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(95), torch.ones(5)]).unsqueeze(1)
    loader, _ = _make({"batch_size": 10})(torch.randn(100, 4), y)
    counts = ns["label_counts"](loader, 2)
    assert counts.tolist() == [95.0, 5.0]
    assert (counts[0] / counts[1]).item() == pytest.approx(19.0)
    assert len(ns["train"](nn.Linear(4, 1), loader)["train_loss"]) == 1


def test_class_weights_compose_with_label_smoothing_and_gate_on_the_loss():
    both = _code({"class_weights": True, "label_smoothing": 0.1})
    assert "nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight)" in both
    # A loss that takes no such argument emits nothing (the metric-spec rule).
    assert "weight=weight" not in _code({"loss": "MSELoss", "class_weights": True})


# --- evaluation on the held-out test split ------------------------------------

def test_generate_eval_reports_loss_metric_and_n():
    from lamplighter.backend.codegen import generate_eval

    src = generate_eval(Graph(), {"device": "cpu", "metric": "accuracy"})
    assert "def evaluate(model, loader, *, device='cpu'):" in src
    assert "model.eval()" in src and "with torch.no_grad():" in src
    assert "opt" not in src and "backward" not in src  # nothing trains here
    ns: dict = {}
    exec(src, ns)  # noqa: S102
    torch.manual_seed(0)
    X, y = torch.randn(24, 4), torch.randint(0, 3, (24,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    result = ns["evaluate"](nn.Linear(4, 3), loader)
    assert set(result) == {"test_loss", "n", "test_acc"}
    assert result["n"] == 24  # a score without its n isn't a result
    # The reported loss is the real one, computable by hand.
    model = nn.Linear(4, 3)
    torch.manual_seed(0)
    with torch.no_grad():
        expected = nn.CrossEntropyLoss()(model(X), y).item()
    assert ns["evaluate"](model, loader)["test_loss"] == pytest.approx(expected, rel=1e-5)


def test_eval_uses_the_plain_objective_not_the_training_devices():
    # Label smoothing regularizes and class weights rebalance — both are
    # training-time devices. A test number has to mean the same thing across
    # runs that used them differently, so evaluate() drops both.
    from lamplighter.backend.codegen import generate_eval

    src = generate_eval(Graph(), {
        "device": "cpu", "loss": "CrossEntropyLoss", "label_smoothing": 0.2, "class_weights": True})
    assert "loss_fn = nn.CrossEntropyLoss()" in src
    assert "label_smoothing" not in src and "weight=" not in src and "label_counts" not in src


def test_eval_metric_gates_on_the_loss_like_everywhere_else():
    from lamplighter.backend.codegen import generate_eval

    assert "test_acc" not in generate_eval(Graph(), {"metric": "accuracy", "loss": "MSELoss"})
    assert "test_mae" in generate_eval(Graph(), {"metric": "mae", "loss": "MSELoss"})
