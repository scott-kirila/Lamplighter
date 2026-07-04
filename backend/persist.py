"""Autosave of the editor design to disk.

The design lives in kernel memory (backend/state.py), so a kernel restart with
no browser tab open used to lose it. This module writes it through to a
per-project file on every mutation and seeds an empty backend from it at session
start — durability you never think about, not a document model (named design
files can build on this later).

The on-disk format is versioned: a v2 file wraps the whole :class:`Project`
(``{"version": 2, "project": {...}}``). A v1 file — a bare graph
(``{"nodes": [...], ...}``) written by an earlier release — is still read and
upgraded to a one-model project via ``project_from_graph``, so an existing
``.lamplighter/graph.json`` keeps working across the schema change.

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

from .schema import Project, project_from_graph
from .schema import Graph as _Graph

_path: Path | None = None


def configure(path: Path | str | None) -> None:
    """Point the autosave at a file; None disables it (the default)."""
    global _path
    _path = Path(path) if path is not None else None


def enable(path: Path | str) -> None:
    """The session-start hook: configure the autosave AND, when the backend
    holds no design (fresh kernel), seed it with the saved one. A backend that
    already has a design (e.g. re-seeded by a still-open tab) wins — it is at
    least as fresh as the file."""
    configure(path)
    from . import state

    if state.get_project() is None:
        saved = load()
        if saved is not None:
            state.set_project(saved)


def save(project: Project) -> None:
    """Write-through, atomically (temp file + rename) so a kernel killed
    mid-write can't corrupt the design. Never raises into the edit path — a
    full disk shouldn't take down shape inference."""
    if _path is None:
        return
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _path.with_suffix(".tmp")
        payload = {"version": 2, "project": project.model_dump()}
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, _path)
    except Exception as exc:
        warnings.warn(f"could not autosave the design to {_path}: {exc}", stacklevel=2)


def load() -> Project | None:
    """The saved project, or None. A missing, corrupt, or schema-incompatible
    file warns and returns None — never fatal; the worst case is starting blank,
    exactly as without autosave. A v1 (bare-graph) file is upgraded to a
    one-model project."""
    if _path is None or not _path.exists():
        return None
    try:
        raw = json.loads(_path.read_text())
        if isinstance(raw, dict) and "project" in raw:
            return Project.model_validate(raw["project"])
        # v1: a bare graph written by an earlier release.
        return project_from_graph(_Graph.model_validate(raw))
    except Exception as exc:
        warnings.warn(f"ignoring the saved design at {_path} ({exc})", stacklevel=2)
        return None
