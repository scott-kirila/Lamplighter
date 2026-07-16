"""Notebook control + client for Lamplighter.

Drive the whole thing from a Jupyter cell::

    import lamplighter
    sess = lamplighter.start()          # serve the app + open the editor in your browser
    model = lamplighter.build_model()   # live nn.Module from whatever is on the canvas
    lamplighter.open_editor()           # reopen the tab if you closed it (work is restored)
    lamplighter.stop()                  # tear the session down

No file juggling: the model tracks the live editor graph. Re-run a cell after
editing the canvas to pick up changes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"


class LamplighterError(RuntimeError):
    """Raised when the backend is unreachable or the graph can't be built."""


def _resolve(base_url: str | None) -> str:
    if base_url is not None:
        return base_url
    # Fall back to the running session's URL, then the conventional default.
    try:
        from .session import current

        sess = current()
        if sess is not None and sess.is_running():
            return sess.url
    except Exception:
        pass
    return DEFAULT_URL


def _get(path: str, base_url: str | None) -> Any:
    url = f"{_resolve(base_url).rstrip('/')}{path}"
    try:
        with urllib.request.urlopen(url) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail", exc.reason)
        except Exception:
            detail = exc.reason
        raise LamplighterError(f"backend returned {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise LamplighterError(
            f"could not reach Lamplighter — is a session running? (start with lamplighter.start()) "
            f"[{exc.reason}]"
        ) from None


def model_code(base_url: str | None = None) -> str:
    """Return the generated ``nn.Module`` source for the current editor graph."""
    return _get("/api/model/code", base_url)["code"]


def graph(base_url: str | None = None) -> dict[str, Any]:
    """Return the current editor graph as JSON (nodes + edges)."""
    return _get("/api/graph", base_url)


def training_code(base_url: str | None = None) -> str:
    """Return the generated ``train(model, loader)`` source for the current config."""
    return _get("/api/training/code", base_url)["code"]


def data_code(base_url: str | None = None) -> str:
    """Return the generated ``make_dataloaders()`` source for the current data config."""
    return _get("/api/data/code", base_url)["code"]


def build_trainer(base_url: str | None = None):
    """Build the live ``train`` function from the current training config."""
    from backend.codegen import exec_generated

    code = training_code(base_url)
    namespace = exec_generated(code, "<lamplighter-generated-trainer>")
    if "train" not in namespace:
        raise LamplighterError("generated code did not define train")
    return namespace["train"]


def build_dataloaders(base_url: str | None = None):
    """Build the live ``make_dataloaders`` function from the current data config.

    It returns ``(train_loader, val_loader)`` — feed them to the trainer::

        train_loader, val_loader = lamplighter.build_dataloaders()(X, y)
        lamplighter.build_trainer()(model, train_loader, val_loader=val_loader)
    """
    from backend.codegen import exec_generated

    code = data_code(base_url)
    namespace = exec_generated(code, "<lamplighter-generated-dataloaders>")
    if "make_dataloaders" not in namespace:
        raise LamplighterError("generated code did not define make_dataloaders")
    return namespace["make_dataloaders"]


def _model_class(namespace: dict[str, Any]):
    """The model class in a generated module namespace — found by type, so a
    per-model class name (``Generator``) works the same as the classic
    ``GeneratedModel``. The *last* ``nn.Module`` subclass wins: spliced
    Custom-node classes precede the model class, which codegen writes last."""
    import torch.nn as nn

    found = None
    for value in namespace.values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            found = value
    if found is None:
        raise LamplighterError("generated code did not define a model class")
    return found


def build_model(base_url: str | None = None):
    """Build and instantiate the live model as a ``torch.nn.Module``."""
    from backend.codegen import exec_generated

    code = model_code(base_url)
    return _model_class(exec_generated(code, "<lamplighter-generated-model>"))()


def load_checkpoint(path: str, best: bool = False, model: str | None = None):
    """Rebuild a trained model from a checkpoint saved by ``sess.save_checkpoint()``
    (or the app's weights download) — no session or graph needed. The checkpoint
    is self-contained: the model is reconstructed from the generated source
    embedded in its snapshot, then the trained weights are loaded.

    ``best=True`` loads the weights from the epoch with the lowest validation
    loss instead of the final ones (available when the run had validation).

    A multi-model checkpoint (e.g. a GAN) holds several models; pass
    ``model="generator"`` (a role name) to pick one. With a single model the
    argument is unnecessary.

    Returns ``(model, snapshot)`` — the model in eval mode on CPU, and the run's
    reproducibility record (seed, configs, sources, …).
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    snapshot = checkpoint["snapshot"]

    roles = list(checkpoint["state_dicts"])
    if model is None:
        if len(roles) != 1:
            raise LamplighterError(
                f"this checkpoint holds several models ({', '.join(roles)}) — "
                f"pass model=<name>, e.g. load_checkpoint(path, model='{roles[0]}')"
            )
        model = roles[0]
    if model not in checkpoint["state_dicts"]:
        raise LamplighterError(f"no model '{model}' here (models: {', '.join(roles)})")
    state = checkpoint["state_dicts"][model]
    if best:
        state = checkpoint.get("best_state_dict")
        if state is None:
            raise LamplighterError(
                "this checkpoint has no best-epoch weights — the run had no validation"
            )
    source = snapshot["sources"]["models"][model]

    from backend.codegen import exec_generated

    rebuilt = _model_class(exec_generated(source, "<lamplighter-checkpoint-model>"))()
    rebuilt.load_state_dict(state)
    return rebuilt.eval(), snapshot


from .session import Lamplighter, Session, current, open_editor, start, status, stop  # noqa: E402

__all__ = [
    "Lamplighter",
    "start",
    "stop",
    "open_editor",
    "status",
    "current",
    "Session",
    "build_model",
    "load_checkpoint",
    "model_code",
    "build_trainer",
    "training_code",
    "build_dataloaders",
    "data_code",
    "graph",
    "LamplighterError",
    "DEFAULT_URL",
]
