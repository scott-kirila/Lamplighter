"""Notebook client for a running Scorch backend.

Keep the editor open in your browser, then in a Jupyter cell::

    import scorch
    model = scorch.build_model()   # live nn.Module from the editor graph
    print(scorch.model_code())     # inspect the generated source

No file juggling — the model tracks whatever is on the canvas right now.
Re-run the cell after editing to pick up changes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"


class ScorchError(RuntimeError):
    """Raised when the backend is unreachable or the graph can't be built."""


def _get(path: str, base_url: str) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
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
            f"could not reach Scorch at {base_url} — is `python main.py` running? ({exc.reason})"
        ) from None


def model_code(base_url: str = DEFAULT_URL) -> str:
    """Return the generated ``nn.Module`` source for the current editor graph."""
    return _get("/api/model/code", base_url)["code"]


def graph(base_url: str = DEFAULT_URL) -> dict[str, Any]:
    """Return the current editor graph as JSON (nodes + edges)."""
    return _get("/api/graph", base_url)


def build_model(base_url: str = DEFAULT_URL):
    """Build and instantiate the live model as a ``torch.nn.Module``."""
    code = model_code(base_url)
    namespace: dict[str, Any] = {}
    exec(compile(code, "<scorch-generated-model>", "exec"), namespace)
    if "GeneratedModel" not in namespace:
        raise ScorchError("generated code did not define GeneratedModel")
    return namespace["GeneratedModel"]()


__all__ = ["build_model", "model_code", "graph", "ScorchError", "DEFAULT_URL"]
