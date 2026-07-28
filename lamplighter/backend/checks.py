"""Pre-flight checks for any ``nn.Module`` + the data you'd train it on.

``lamplighter.check(model, data, loss=...)`` is the headless half of the
pre-flight panel: no session, no canvas, no graph. Where the canvas checks
read a Graph, these read the live objects — the model's real modules (via
``named_modules()``) and its real behaviour (one forward pass on a real
batch, in eval mode, under ``no_grad``). That makes them run on anything
that is an ``nn.Module``: hand-written, generated, imported, HuggingFace.

Behavioural probes are preferred over structural walks wherever they are
stronger: a functional ``F.softmax`` in ``forward()`` is invisible to any
module walk but obvious from "do the output rows sum to 1".

Every check is a row: ``{"level": "ok" | "warn" | "error", "title",
"detail"}`` — the same shape the canvas checklist renders, so the two
surfaces can never disagree about what a finding looks like.

This module is deliberately self-contained: stdlib + torch (numpy optional),
no imports from the rest of the package, so it can be loaded as a single
file by path (``importlib.util.spec_from_file_location``). The MCP server
does exactly that inside a bare subprocess — the environment being checked
needs torch, not a lamplighter install. Shared helpers live here and
``diagnose`` imports them, keeping one definition per check.
"""
from __future__ import annotations

from typing import Any

_CLASSIFICATION_LOSSES = ("CrossEntropyLoss", "NLLLoss")

# Losses that accept a per-class weight= argument (the imbalance remedy).
# codegen and diagnose import this — the single source of truth.
_WEIGHTABLE_LOSSES = ("CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss")

# Losses whose target contract is known well enough to check. Anything else
# (a custom callable, an exotic loss) gets "not checked", never a guess.
_REGRESSION_LOSSES = ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss",
                      "BCEWithLogitsLoss", "BCELoss")
_KNOWN_LOSSES = _CLASSIFICATION_LOSSES + _REGRESSION_LOSSES

# torch.nn.functional spellings of the same losses, so passing F.cross_entropy
# works the same as nn.CrossEntropyLoss().
_FUNCTIONAL_LOSSES = {
    "cross_entropy": "CrossEntropyLoss",
    "nll_loss": "NLLLoss",
    "mse_loss": "MSELoss",
    "l1_loss": "L1Loss",
    "smooth_l1_loss": "SmoothL1Loss",
    "huber_loss": "HuberLoss",
    "binary_cross_entropy_with_logits": "BCEWithLogitsLoss",
    "binary_cross_entropy": "BCELoss",
}

# Flag an imbalance once the biggest class outnumbers the smallest by this
# much — the point where an unweighted model starts winning by predicting the
# majority. A judgement call, deliberately loose: it's advice, not a blocker.
_IMBALANCE_RATIO = 3.0

# Samples read for the forward probe when the data isn't already batched.
_PROBE_SAMPLES = 8


def _row(level: str, title: str, detail: str = "") -> dict[str, str]:
    return {"level": level, "title": title, "detail": detail}


def _fmt(dims: list[int]) -> str:
    return "(" + ", ".join(str(d) for d in dims) + ")"


def _arraylike_spec(x: Any) -> tuple[list[int], bool] | None:
    """(shape, is_integer_dtype) for a torch Tensor or numpy array, else None."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            ints = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
            return list(x.shape), x.dtype in ints
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return list(x.shape), bool(np.issubdtype(x.dtype, np.integer))
    except Exception:
        pass
    return None


def _class_counts(y: Any) -> list[int] | None:
    """Per-class sample counts for a class-like target — integer labels, or a
    float target that only holds 0/1 (what BCEWithLogits takes). None for
    anything else (a regression target has no classes to count)."""
    try:
        import torch

        t = y.flatten()
        if t.dtype.is_floating_point and set(t.unique().tolist()) - {0.0, 1.0}:
            return None
        t = t.long()
        if int(t.min()) < 0:
            return None
        return torch.bincount(t).tolist()
    except Exception:
        return None


def _dataset_size(ds: Any) -> int | None:
    """``len(ds)``, or None for an iterable-style dataset that has no length."""
    try:
        return int(len(ds))
    except (TypeError, AttributeError):
        return None


def _dataset_labels(ds: Any) -> Any | None:
    """The label tensor a dataset already holds, WITHOUT iterating or decoding.

    This deliberately never pulls a batch. ``diagnose`` runs on every edit, so
    touching the data would decode and augment images on each keystroke for an
    ImageFolder, and consuming an ``IterableDataset`` is not even replayable.
    Reading what the dataset *declares* covers the formats people actually
    register — torchvision's ``targets``, ImageFolder, ``TensorDataset``,
    ``Subset`` — and anything else is reported as unknown rather than guessed.
    """
    import torch

    if ds is None:
        return None
    # Subset: index the parent's labels by the subset's own indices, so a
    # random_split's class range is the split's, not the whole dataset's.
    indices, parent = getattr(ds, "indices", None), getattr(ds, "dataset", None)
    if indices is not None and parent is not None:
        inner = _dataset_labels(parent)
        if inner is None:
            return None
        try:
            return inner[torch.as_tensor(list(indices), dtype=torch.long)]
        except Exception:
            return None
    # TensorDataset keeps its tensors; the last is the target by convention.
    tensors = getattr(ds, "tensors", None)
    if tensors is not None and len(tensors) >= 2:
        return tensors[-1]
    for attr in ("targets", "labels"):
        values = getattr(ds, attr, None)
        if values is None:
            continue
        try:
            return values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        except Exception:
            return None
    return None


def _check_class_range(checks: list, y_name: str, y: Any, n_classes: int) -> None:
    """Labels vs. the model's output width — the flagship check. Out-of-range
    labels are DEVICE-DEPENDENT: an IndexError on CPU, an async device-side
    assert on CUDA, and no error at all on MPS — the loss just comes out wrong."""
    try:  # a real read of the label tensor — cheap at notebook scale
        y_max, y_min = int(y.max()), int(y.min())
    except Exception:
        return
    if y_min < 0 or y_max >= n_classes:
        checks.append(_row(
            "error",
            f"'{y_name}' has classes {y_min}…{y_max} but the model outputs {n_classes}",
            "this would crash mid-run — adjust the last layer's out_features",
        ))
    elif n_classes > y_max + 1:
        # Runs fine, but the extra logits can never be a right answer —
        # usually a forgotten out_features default on the last layer.
        checks.append(_row(
            "warn",
            f"the model outputs {n_classes} classes but '{y_name}' only uses {y_min}…{y_max}",
            f"did you mean out_features={y_max + 1} on the last layer?",
        ))
    else:
        checks.append(_row("ok", f"classes {y_min}…{y_max} match the model's {n_classes} outputs"))


def _check_loss_fit(
    checks: list, loss: str, y_name: str, y: Any, y_dims: list[int], y_int: bool,
    model_output: list[int] | None, fix_style: str = "registered",
) -> None:
    """Does the target actually fit the chosen loss (and the model's output)?

    ``fix_style`` picks the wording of the fix, because the finding is
    identical across surfaces but the fix isn't: ``"registered"`` for a tensor
    registered via ``sess.data(...)`` (re-run the call), ``"dataset"`` for
    labels read out of a Dataset/DataLoader (there is no call to re-run), and
    ``"headless"`` for a tensor passed straight to ``check()``.
    """
    if loss == "Custom":
        # A registered loss class: its target contract is the user's business,
        # so no dtype/shape rule applies. Say what IS knowable — the metric
        # specs gate on built-in losses, so none can be reported.
        checks.append(_row(
            "ok", "custom loss — target fit isn't checked",
            "shape/dtype rules are yours; per-epoch metrics report loss only",
        ))
        return
    if loss in _CLASSIFICATION_LOSSES:
        if not y_int:
            fix = {
                "registered": f"e.g. sess.data({y_name}={y_name}.long())",
                "dataset": "cast the dataset's targets to long",
                "headless": f"cast them to long: {y_name}.long()",
            }[fix_style]
            checks.append(_row("error", f"{loss} needs integer class targets but '{y_name}' is float", fix))
            return
        if len(y_dims) != 1:
            detail = ""
            if len(y_dims) == 2 and y_dims[1] == 1:  # the (N, 1) column-vector classic
                detail = {
                    "registered": f"squeeze the extra dim: sess.data({y_name}={y_name}.squeeze(1))",
                    "dataset": "drop the trailing dim from the dataset's targets",
                    "headless": f"squeeze the extra dim: {y_name}.squeeze(1)",
                }[fix_style]
            checks.append(_row("error", f"{loss} expects 1-D class targets but '{y_name}' is {_fmt(y_dims)}", detail))
            return
        if model_output is not None and len(model_output) == 2:
            _check_class_range(checks, y_name, y, model_output[-1])
    else:
        if y_int:
            fix = {
                "registered": f"e.g. sess.data({y_name}={y_name}.float())",
                "dataset": "cast the dataset's targets to float",
                "headless": f"cast them to float: {y_name}.float()",
            }[fix_style]
            checks.append(_row("error", f"{loss} needs float targets but '{y_name}' is integer", fix))
        elif model_output is not None and y_dims[1:] != model_output[1:]:
            checks.append(_row("warn", f"'{y_name}' sample {_fmt(y_dims[1:])} vs model output {_fmt(model_output[1:])}",
                               f"{loss} may broadcast unexpectedly"))


class CheckReport:
    """The result of :func:`check`: an ordered list of finding rows plus the
    one verdict that matters (``.ok`` — no errors). Prints as the checklist;
    ``to_dict()`` is the JSON the MCP tool returns."""

    _MARKS = {"ok": "✓", "warn": "⚠", "error": "✗"}

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = list(rows)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [r for r in self.rows if r["level"] == "error"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [r for r in self.rows if r["level"] == "warn"]

    @property
    def ok(self) -> bool:
        """True when nothing rose to an error — the "safe to train" verdict."""
        return not self.errors

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "checks": self.rows,
        }

    def __repr__(self) -> str:
        def plural(n: int, word: str) -> str:
            return f"{n} {word}{'' if n == 1 else 's'}"

        n_ok = len(self.rows) - len(self.errors) - len(self.warnings)
        head = f"{'✓' if self.ok else '✗'} {plural(len(self.errors), 'error')}, " \
               f"{plural(len(self.warnings), 'warning')}, {n_ok} ok"
        lines = [head]
        for r in self.rows:
            lines.append(f"  {self._MARKS.get(r['level'], '?')} {r['title']}")
            if r.get("detail"):
                lines.append(f"      {r['detail']}")
        return "\n".join(lines)


def _loss_name(loss: Any) -> str | None:
    """The canonical loss-class name for whatever was passed: an instance
    (``nn.CrossEntropyLoss()``), the class, a ``torch.nn.functional`` function,
    or the name itself. None only when ``loss`` is None."""
    if loss is None:
        return None
    if isinstance(loss, str):
        return loss
    if isinstance(loss, type):
        return loss.__name__
    fn_name = getattr(loss, "__name__", None)
    if fn_name is not None:
        return _FUNCTIONAL_LOSSES.get(fn_name, fn_name)
    return type(loss).__name__


def _coerce_tensor(checks: list, name: str, value: Any) -> Any | None:
    """The value as a torch Tensor — flagging (not hiding) a numpy array, which
    the training loop would also refuse. Continues checking with the converted
    tensor so one wrong dtype doesn't hide every other finding."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            checks.append(_row("error", f"'{name}' is a numpy array",
                               f"convert it with torch.from_numpy({name})"))
            return torch.from_numpy(value)
    except ImportError:
        pass
    checks.append(_row("error", f"'{name}' isn't a tensor ({type(value).__name__})",
                       "pass a torch.Tensor, a Dataset, or a DataLoader"))
    return None


def _first_batch(checks: list, loader: Any, dataset: Any) -> Any | None:
    """One real batch from the loader, reported (not raised) when it can't."""
    from torch.utils.data import IterableDataset

    if isinstance(dataset, IterableDataset):
        checks.append(_row(
            "warn", "reading one batch from an IterableDataset",
            "if it streams from a one-shot source, the probe consumes those items",
        ))
    try:
        return next(iter(loader))
    except StopIteration:
        checks.append(_row("error", "the DataLoader yielded no batches",
                           "an empty loader trains nothing — check its dataset and drop_last"))
    except Exception as exc:
        checks.append(_row("error", f"the DataLoader failed to yield a batch: {type(exc).__name__}",
                           str(exc)[:300]))
    return None


def _collate_probe(checks: list, dataset: Any, n: int | None) -> Any | None:
    """A small batch collated from a bare dataset — what a default DataLoader
    would feed the model."""
    from torch.utils.data import IterableDataset, default_collate

    try:
        if isinstance(dataset, IterableDataset):
            checks.append(_row(
                "warn", f"reading {_PROBE_SAMPLES} samples from an IterableDataset",
                "if it streams from a one-shot source, the probe consumes those items",
            ))
            from itertools import islice

            samples = list(islice(iter(dataset), _PROBE_SAMPLES))
        else:
            samples = [dataset[i] for i in range(min(n or 1, _PROBE_SAMPLES))]
        if not samples:
            checks.append(_row("error", "the dataset yielded no samples"))
            return None
        return default_collate(samples)
    except Exception as exc:
        checks.append(_row("error", f"couldn't read samples from the dataset: {type(exc).__name__}",
                           str(exc)[:300]))
        return None


def _split_batch(batch: Any) -> tuple[Any | None, dict | None, Any | None]:
    """(x, kwargs, y) from whatever one batch looks like: a bare tensor, an
    (x, y)-style tuple (first in, last out), or an HF-style dict of tensors
    (fed as keyword arguments; ``labels`` is the target by convention)."""
    import torch

    if isinstance(batch, torch.Tensor):
        return batch, None, None
    if isinstance(batch, dict):
        return None, batch, batch.get("labels")
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0], None, (batch[-1] if len(batch) >= 2 else None)
    return None, None, None


def check(model: Any, data: Any, y: Any = None, *, loss: Any = None,
          batch_size: int | None = None) -> CheckReport:
    """Pre-flight a model against the data it's about to train on.

    Reads the real objects — one forward pass on a real batch, the actual
    label values, the loader's actual arithmetic — and returns a
    :class:`CheckReport` of rows (``ok`` / ``warn`` / ``error``). Catches the
    failures that don't announce themselves: labels out of range for the
    output layer (a mid-run CUDA assert — or *silently wrong loss* on MPS),
    float labels under ``CrossEntropyLoss``, a softmax stacked under
    ``CrossEntropyLoss`` (found behaviourally, so ``F.softmax`` hiding in
    ``forward()`` counts), misaligned X/y, class imbalance, NaN in the
    outputs, and the final-batch-of-1 × BatchNorm crash.

    ``data`` may be a ``DataLoader`` (its ``batch_size``/``drop_last``
    arithmetic is checked too), a ``Dataset``, an ``(X, y)`` pair, a bare
    tensor (targets in ``y=``), or an HF-style dict of tensors. ``loss`` may
    be a loss instance, class, ``torch.nn.functional`` function, or name;
    without it the loss↔target checks are skipped and say so. ``batch_size``
    feeds the batch arithmetic when the data isn't already a loader.

    The model is run once in eval mode under ``no_grad`` (training mode is
    restored), on the model's own device. Nothing is copied or written.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if data is None:
        raise TypeError("data must be a DataLoader, Dataset, (X, y) pair, tensor, or dict of tensors")

    checks: list[dict[str, str]] = []

    # -- normalize the data argument -------------------------------------------
    X = y_full = loader = dataset = kw_full = None
    if isinstance(data, DataLoader):
        loader = data
        dataset = getattr(data, "dataset", None)
    elif isinstance(data, Dataset):
        dataset = data
    elif isinstance(data, dict):
        kw_full = data
        y_full = data.get("labels")
    elif isinstance(data, (tuple, list)):
        if len(data) != 2:
            raise TypeError(f"a data tuple must be (X, y), got {len(data)} elements")
        X, y_full = data
    else:
        X = data
    if y is not None:
        if y_full is not None:
            raise TypeError("targets were passed twice — in `data` and as y=")
        y_full = y

    if X is not None:
        X = _coerce_tensor(checks, "X", X)
    y_scope = "full"
    if y_full is not None:
        y_full = _coerce_tensor(checks, "y", y_full)

    # -- sample count -----------------------------------------------------------
    n: int | None = None
    if X is not None:
        if X.dim() == 0:
            checks.append(_row("error", "'X' is a scalar tensor"))
            X = None
        else:
            n = X.shape[0]
    elif dataset is not None:
        n = _dataset_size(dataset)
    elif kw_full is not None:
        probe = y_full if y_full is not None else next(
            (v for v in kw_full.values() if isinstance(v, torch.Tensor)), None)
        n = probe.shape[0] if probe is not None and probe.dim() else None
    if n == 0:
        checks.append(_row("error", "0 samples — there is nothing to train on"))
        return CheckReport(checks)

    # -- model + data facts, so the reader can confirm we're checking the right
    # objects before trusting any verdict below.
    p = next(iter(model.parameters()), None)
    device = p.device if p is not None else torch.device("cpu")
    n_params = sum(param.numel() for param in model.parameters())
    checks.append(_row("ok", f"{type(model).__name__}: {n_params:,} parameters on {device}"))
    if loader is not None:
        b = getattr(loader, "batch_size", None)
        sized = f" of {n} samples" if n is not None else ""
        batched = f", batch_size {b}" if b else ""
        checks.append(_row("ok", f"data: a DataLoader{sized}{batched}"))
    elif dataset is not None:
        sized = f" of {n} samples" if n is not None else " (no length)"
        checks.append(_row("ok", f"data: a {type(dataset).__name__}{sized}"))
    elif X is not None:
        checks.append(_row("ok", f"data: {n} samples of {_fmt(list(X.shape[1:]))}"))

    loss_name = _loss_name(loss)
    known_loss = loss_name if loss_name in _KNOWN_LOSSES else None
    if loss is None:
        checks.append(_row("warn", "no loss given — loss ↔ target fit isn't checked",
                           "pass loss=... (an instance, the class, or its name) to check it"))
    elif known_loss is None:
        checks.append(_row("ok", f"loss '{loss_name}' isn't a torch built-in — target fit isn't checked",
                           "its shape/dtype contract is yours"))

    # -- X ↔ y alignment (tensor data owns its pairing; a dataset pairs internally)
    if X is not None and y_full is not None and y_full.dim() and n is not None:
        if y_full.shape[0] != n:
            checks.append(_row("error", f"'X' has {n} samples but 'y' has {y_full.shape[0]}",
                               "they must align row-for-row"))
        else:
            checks.append(_row("ok", f"{n} samples — X and y aligned"))

    # -- labels: prefer what covers the whole dataset; fall back to one batch --
    if y_full is None and dataset is not None:
        y_full = _dataset_labels(dataset)

    # -- the forward probe ------------------------------------------------------
    x_probe = kw_probe = None
    if loader is not None:
        batch = _first_batch(checks, loader, dataset)
        if batch is not None:
            x_probe, kw_probe, y_batch = _split_batch(batch)
            if y_full is None and y_batch is not None and isinstance(y_batch, torch.Tensor):
                y_full, y_scope = y_batch, "batch"
    elif dataset is not None:
        batch = _collate_probe(checks, dataset, n)
        if batch is not None:
            x_probe, kw_probe, y_batch = _split_batch(batch)
            if y_full is None and y_batch is not None and isinstance(y_batch, torch.Tensor):
                y_full, y_scope = y_batch, "batch"
    elif kw_full is not None:
        k = min(n or _PROBE_SAMPLES, _PROBE_SAMPLES)
        kw_probe = {key: (v[:k] if isinstance(v, torch.Tensor) and v.dim() else v)
                    for key, v in kw_full.items()}
    elif X is not None:
        x_probe = X[: min(n, batch_size or 32)]

    output: Any = None
    if x_probe is not None or kw_probe is not None:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if kw_probe is not None:
                    moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in kw_probe.items()}
                    out = model(**moved)
                else:
                    out = model(x_probe.to(device))
        except Exception as exc:
            checks.append(_row(
                "error", f"the model can't consume a real batch: {type(exc).__name__}",
                f"{str(exc)[:300]} — this is the error the first training step would hit",
            ))
            out = None
        finally:
            if was_training:
                model.train()
        if out is not None:
            # HF models return an output object with .logits; some models
            # return tuples or dicts — check the head, say so if there's
            # nothing usable.
            out = getattr(out, "logits", out)
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]
            if isinstance(out, (tuple, list)) and out:
                out = out[0]
            if isinstance(out, torch.Tensor):
                output = out
                fed = _fmt(list(x_probe.shape)) if x_probe is not None else "a dict batch"
                checks.append(_row("ok", f"forward pass: {fed} → {_fmt(list(output.shape))}",
                                   "run on a real batch, in eval mode, no gradients"))
            else:
                checks.append(_row("warn", f"the model returned a {type(out).__name__}, not a tensor",
                                   "output-side checks (activation, class range) are skipped"))

    output_dims = list(output.shape) if output is not None else None

    # -- what the output says, behaviourally -----------------------------------
    if output is not None and output.dim() >= 1:
        if not torch.isfinite(output).all():
            checks.append(_row("error", "the model outputs NaN/Inf on a real batch",
                               "before any training step — check init, normalization, and divisions in forward()"))
        # A batch of B samples should come out as B rows. B*T rows means a
        # view/reshape in forward() is folding the batch dim; (T, B, …) means a
        # seq-first module met a batch-first pipeline.
        B = x_probe.shape[0] if x_probe is not None and x_probe.dim() else None
        if B is not None and B >= 2 and output.shape[0] != B:
            if output.dim() >= 2 and output.shape[1] == B:
                checks.append(_row(
                    "warn", f"output is {_fmt(output_dims)} — batch dim second, not first",
                    "looks seq-first (T, B, …): if the loader yields batch-first, an RNN/attention "
                    "module with batch_first=False is transposing samples and positions",
                ))
            else:
                checks.append(_row(
                    "error", f"a batch of {B} samples came out as {output.shape[0]} rows",
                    "a view/reshape in forward() is probably folding the batch dimension — "
                    "use x.view(x.size(0), -1) style reshapes",
                ))

    # The classic logits-vs-probabilities footgun, read off the real outputs —
    # so a functional F.softmax (invisible to any module walk) is caught too.
    if (output is not None and known_loss in _CLASSIFICATION_LOSSES
            and output.dtype.is_floating_point and output.dim() in (2, 3)
            and output.shape[-1] >= 2 and output.numel() // output.shape[-1] >= 2):
        flat = output.float().reshape(-1, output.shape[-1])
        ones = flat.new_ones(flat.shape[0])  # same device as the output
        is_probs = bool(flat.min() >= 0) and bool(flat.max() <= 1 + 1e-6) \
            and torch.allclose(flat.sum(-1), ones, atol=1e-4)
        is_logp = bool(flat.max() <= 1e-6) \
            and torch.allclose(flat.exp().sum(-1), ones, atol=1e-4)
        if known_loss == "CrossEntropyLoss":
            if is_probs:
                checks.append(_row(
                    "error", "the model outputs probabilities but CrossEntropyLoss expects raw logits",
                    "every output row is ≥0 and sums to 1 on a real batch — remove the final "
                    "softmax (CrossEntropyLoss applies log-softmax itself); training on "
                    "double-softmaxed outputs flattens gradients and caps the loss",
                ))
            elif is_logp:
                checks.append(_row(
                    "warn", "the model outputs log-probabilities under CrossEntropyLoss",
                    "numerically harmless (log-softmax is idempotent) but redundant — "
                    "feed raw logits, or switch the loss to NLLLoss",
                ))
            else:
                checks.append(_row("ok", "outputs look like raw logits — what CrossEntropyLoss expects"))
        else:  # NLLLoss
            if is_logp:
                checks.append(_row("ok", "log-probabilities → NLLLoss: matched"))
            elif is_probs:
                checks.append(_row(
                    "error", "the model outputs probabilities but NLLLoss expects log-probabilities",
                    "every output row sums to 1 on a real batch — use LogSoftmax, not Softmax "
                    "(or switch to CrossEntropyLoss on raw logits)",
                ))
            else:
                checks.append(_row(
                    "warn", "the model doesn't output log-probabilities but NLLLoss expects them",
                    "add a LogSoftmax head, or switch to CrossEntropyLoss on raw logits",
                ))

    # -- loss ↔ target fit, including the class-range check --------------------
    if y_full is not None:
        spec = _arraylike_spec(y_full)
        if spec is not None:
            y_dims, y_int = spec
            if y_scope == "batch":
                # One batch can prove labels wrong, never prove them right —
                # so the in-range verdict is scoped, not overclaimed.
                if known_loss in _CLASSIFICATION_LOSSES and not y_int:
                    _check_loss_fit(checks, known_loss, "y", y_full, y_dims, y_int, None, "dataset")
                elif output_dims is not None and len(output_dims) in (2, 3) and y_int:
                    C = output_dims[-1]
                    try:
                        y_min, y_max = int(y_full.min()), int(y_full.max())
                    except Exception:
                        y_min = y_max = None
                    if y_min is not None and (y_min < 0 or y_max >= C):
                        checks.append(_row(
                            "error", f"first-batch labels run {y_min}…{y_max} but the model outputs {C}",
                            "this would crash mid-run — adjust the last layer's out_features",
                        ))
                    elif y_min is not None:
                        checks.append(_row(
                            "ok", f"first-batch labels {y_min}…{y_max} fit the model's {C} outputs",
                            "only one batch was read — pass y= (the full labels) to check them all",
                        ))
            elif known_loss is not None and output_dims is not None and len(output_dims) == 3 \
                    and known_loss in _CLASSIFICATION_LOSSES:
                # Sequence outputs (B, T, C): targets are (B, T), so the 1-D
                # rule doesn't apply — check dtype and range against C.
                if not y_int:
                    _check_loss_fit(checks, known_loss, "y", y_full, y_dims, y_int, None, "headless")
                else:
                    _check_class_range(checks, "y", y_full, output_dims[-1])
            elif known_loss is not None:
                out2 = output_dims if output_dims is not None and len(output_dims) == 2 else None
                _check_loss_fit(checks, known_loss, "y", y_full, y_dims, y_int, out2, "headless")
            elif loss is None and y_int and len(y_dims) == 1 \
                    and output_dims is not None and len(output_dims) == 2:
                # No loss named, but integer 1-D labels against (N, C) outputs
                # is classification in any loop — the range must hold anyway.
                _check_class_range(checks, "y", y_full, output_dims[-1])
            if y_scope == "full":
                _check_imbalance_headless(checks, known_loss, "y", y_full)
    elif known_loss is not None and (loader is not None or dataset is not None):
        checks.append(_row(
            "warn", "no labels found — loss ↔ target fit isn't checked",
            "the class-range, dtype and imbalance checks didn't run — pass y= (the label tensor)",
        ))

    # -- batch arithmetic: the crash that is fully predictable from four numbers
    _check_batch_arithmetic(checks, model, n, loader, batch_size)

    return CheckReport(checks)


def _check_imbalance_headless(checks: list, loss_name: str | None, y_name: str, y: Any) -> None:
    """Class balance, with remedies phrased for code (not the app's forms)."""
    counts = _class_counts(y)
    if counts is None:
        return
    present = [c for c in counts if c > 0]
    if len(present) < 2:
        return
    ratio = max(present) / min(present)
    if ratio < _IMBALANCE_RATIO:
        return
    spread = ", ".join(f"{i}: {c}" for i, c in enumerate(counts) if c)
    remedies = ["a WeightedRandomSampler on the DataLoader"]
    if loss_name in _WEIGHTABLE_LOSSES:
        remedies.insert(0, f"weight= on {loss_name}")
    checks.append(_row(
        "warn", f"classes are imbalanced ({ratio:.0f}:1)",
        f"{spread} — consider {' or '.join(remedies)}",
    ))


def _check_batch_arithmetic(
    checks: list, model: Any, n: int | None, loader: Any, batch_size: int | None
) -> None:
    """The BatchNorm × batch-of-1 crash and drop_last waste — deterministic
    from n, batch_size and drop_last, with BatchNorm found by isinstance walk
    (any ``_BatchNorm`` subclass, however deeply nested)."""
    from torch.nn.modules.batchnorm import _BatchNorm

    bn_types = sorted({type(m).__name__ for m in model.modules() if isinstance(m, _BatchNorm)})

    if loader is not None:
        batch = getattr(loader, "batch_size", None)  # None under a custom batch_sampler
        drop_last = bool(getattr(loader, "drop_last", False))
    else:
        batch, drop_last = batch_size, False

    if bn_types and batch is None:
        checks.append(_row(
            "warn", f"{'/'.join(bn_types)} in the model, but no batch size to check it against",
            "pass batch_size= to predict the final-batch-of-1 crash",
        ))
        return
    if not batch or batch < 1 or n is None:
        return

    bn = "/".join(bn_types)
    ragged = n % batch
    if bn_types:
        if batch == 1:
            checks.append(_row("error", f"batch_size 1 with {bn} in the model",
                               "BatchNorm needs more than 1 sample per training batch"))
        elif ragged == 1 and not drop_last:
            checks.append(_row(
                "error", f"the final batch has 1 sample and the model contains {bn}",
                f"{n} % {batch} = 1 — this crashes in training; pass drop_last=True "
                "to the DataLoader or change its batch_size",
            ))
    if drop_last and ragged and ragged / n >= 0.25:
        checks.append(_row("warn", f"drop_last discards {ragged} of {n} samples every epoch",
                           "the ragged final batch is a big share of your data"))
    if batch > n:
        checks.append(_row("warn", f"batch_size {batch} exceeds the {n} samples",
                           "every epoch is a single batch"))
