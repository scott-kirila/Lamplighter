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
    assert "torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0001)" in code


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
