"""Autosave of the editor graph to disk.

The graph lives in kernel memory (backend/state.py), so a kernel restart with
no browser tab open used to lose the design. This module writes the graph
through to a per-project file on every mutation and seeds an empty backend
from it at session start — durability you never think about, not a document
model (named design files can build on this later).

Disabled by default: nothing writes unless a path is configured, which
``lamplighter.start(persist=...)`` does (default ``.lamplighter/graph.json``
in the notebook's working directory). Tests and bare TestClient apps therefore
never touch the filesystem.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from .schema import Graph

_path: Path | None = None


def configure(path: Path | str | None) -> None:
    """Point the autosave at a file; None disables it (the default)."""
    global _path
    _path = Path(path) if path is not None else None


def enable(path: Path | str) -> None:
    """The session-start hook: configure the autosave AND, when the backend
    holds no graph (fresh kernel), seed it with the saved design. A backend
    that already has a graph (e.g. re-seeded by a still-open tab) wins — it is
    at least as fresh as the file."""
    configure(path)
    from . import state

    if state.get_graph() is None:
        saved = load()
        if saved is not None:
            state.set_graph(saved)


def save(graph: Graph) -> None:
    """Write-through, atomically (temp file + rename) so a kernel killed
    mid-write can't corrupt the design. Never raises into the edit path — a
    full disk shouldn't take down shape inference."""
    if _path is None:
        return
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _path.with_suffix(".tmp")
        tmp.write_text(json.dumps(graph.model_dump(), indent=1))
        os.replace(tmp, _path)
    except Exception as exc:
        warnings.warn(f"could not autosave the graph to {_path}: {exc}", stacklevel=2)


def load() -> Graph | None:
    """The saved graph, or None. A missing, corrupt, or schema-incompatible
    file warns and returns None — never fatal; the worst case is starting
    blank, exactly as without autosave."""
    if _path is None or not _path.exists():
        return None
    try:
        return Graph.model_validate(json.loads(_path.read_text()))
    except Exception as exc:
        warnings.warn(f"ignoring the saved graph at {_path} ({exc})", stacklevel=2)
        return None
