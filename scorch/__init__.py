"""Notebook control + client for Scorch.

Drive the whole thing from a Jupyter cell::

    import scorch
    sess = scorch.start()          # serve the app + open the editor in your browser
    model = scorch.build_model()   # live nn.Module from whatever is on the canvas
    scorch.open_editor()           # reopen the tab if you closed it (work is restored)
    scorch.stop()                  # tear the session down

No file juggling: the model tracks the live editor graph. Re-run a cell after
editing the canvas to pick up changes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"


class ScorchError(RuntimeError):
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
        raise ScorchError(f"backend returned {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ScorchError(
            f"could not reach Scorch — is a session running? (start with scorch.start()) "
            f"[{exc.reason}]"
        ) from None


def model_code(base_url: str | None = None) -> str:
    """Return the generated ``nn.Module`` source for the current editor graph."""
    return _get("/api/model/code", base_url)["code"]


def graph(base_url: str | None = None) -> dict[str, Any]:
    """Return the current editor graph as JSON (nodes + edges)."""
    return _get("/api/graph", base_url)


def build_model(base_url: str | None = None):
    """Build and instantiate the live model as a ``torch.nn.Module``."""
    code = model_code(base_url)
    namespace: dict[str, Any] = {}
    exec(compile(code, "<scorch-generated-model>", "exec"), namespace)
    if "GeneratedModel" not in namespace:
        raise ScorchError("generated code did not define GeneratedModel")
    return namespace["GeneratedModel"]()


from .session import Session, current, open_editor, start, status, stop  # noqa: E402

__all__ = [
    "start",
    "stop",
    "open_editor",
    "status",
    "current",
    "Session",
    "build_model",
    "model_code",
    "graph",
    "ScorchError",
    "DEFAULT_URL",
]
