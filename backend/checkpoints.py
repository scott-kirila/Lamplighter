"""The session's named checkpoint store.

Saving snapshots the last run's checkpoint (the same self-contained format the
weights download and ``sess.save_checkpoint()`` use) into kernel memory under a
name. Entries can be listed, downloaded, deleted, and restored (repopulating
the run manager as if that run had just finished), and they're what warm-start
resume trains from.

Optionally persistent (``lamplighter.start(persist=...)`` enables it alongside
the design autosave): each entry writes through to
``.lamplighter/checkpoints/<name>-<hash>.pt`` (the torch-saved entry) plus a
light ``.json`` meta sidecar, and a fresh kernel hydrates the *listing* from
the sidecars at session start — weights load lazily, only when an entry is
actually used (restore/resume/download/compare). The in-memory store stays the
source of truth; a kernel restart no longer discards saved runs. Disabled by
default, so tests and bare imports never touch the filesystem.

Kernel-side and server-side code share this module in-process (same pattern as
the data registry). Every mutation pushes the fresh listing to open editor tabs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# name → {"checkpoint": dict | None, "created": str, ["meta", "path"]}.
# checkpoint is None for a hydrated-but-not-yet-loaded entry (its listing row
# rides "meta"; load() materializes from "path" on first use).
_store: dict[str, dict[str, Any]] = {}
_dir: Path | None = None


def configure(directory: Path | str | None) -> None:
    """Point the write-through at a directory; None disables it (the default)."""
    global _dir
    _dir = Path(directory) if directory is not None else None


def enable(directory: Path | str) -> None:
    """The session-start hook: configure the write-through AND hydrate the
    listing from the meta sidecars — lazily (no weights are read here). An
    entry already in the live store wins over its file (it is at least as
    fresh, same rule as the design autosave)."""
    configure(directory)
    assert _dir is not None
    if not _dir.exists():
        return
    hydrated = False
    for sidecar in sorted(_dir.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text())
            name = str(meta["name"])
            pt = sidecar.with_suffix(".pt")
            if name in _store or not pt.exists():
                continue
            _store[name] = {
                "checkpoint": None,
                "created": meta.get("created", ""),
                "meta": meta,
                "path": pt,
            }
            hydrated = True
        except Exception as exc:
            warnings.warn(f"ignoring the saved checkpoint at {sidecar} ({exc})", stacklevel=2)
    if hydrated:
        _push()


def _entry_path(name: str) -> Path:
    """The entry's file stem: a readable sanitized name + a short hash of the
    true name, so any name is filesystem-safe and distinct names never collide."""
    assert _dir is not None
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "checkpoint"
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return _dir / f"{safe}-{digest}.pt"


def _write_through(name: str, entry: dict[str, Any]) -> None:
    """Persist one entry (torch-saved payload + meta sidecar), atomically per
    file. Never raises into the save path — a full disk shouldn't lose the
    in-memory save (persist.py's rule)."""
    if _dir is None:
        return
    try:
        import torch

        _dir.mkdir(parents=True, exist_ok=True)
        path = _entry_path(name)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"name": name, "created": entry["created"], "checkpoint": entry["checkpoint"]}, tmp)
        os.replace(tmp, path)
        meta_tmp = path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(_meta(name, entry)))
        os.replace(meta_tmp, path.with_suffix(".json"))
    except Exception as exc:
        warnings.warn(f"could not persist checkpoint '{name}': {exc}", stacklevel=2)


def _remove_files(name: str) -> None:
    if _dir is None:
        return
    path = _entry_path(name)
    for p in (path, path.with_suffix(".json")):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def _meta(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """A listing row: identity + the numbers that distinguish checkpoints
    (where training stood, how good it was, how to reproduce it). A hydrated
    placeholder answers from its sidecar meta — no weights are read to list."""
    if entry["checkpoint"] is None:
        return dict(entry["meta"])
    checkpoint = entry["checkpoint"]
    snapshot = checkpoint.get("snapshot") or {}
    val = (checkpoint.get("history") or {}).get("val_loss") or []
    plan = (snapshot.get("training") or {}).get("epochs")
    return {
        "name": name,
        "created": entry["created"],
        "epoch": checkpoint.get("epoch"),
        # The run's planned total — epoch < epochs marks an interrupted run,
        # which resume finishes by default.
        "epochs": int(plan) if plan is not None else None,
        "best_epoch": checkpoint.get("best_epoch"),
        "seed": snapshot.get("seed"),
        "val_loss": val[-1] if val else None,
    }


def metas() -> list[dict[str, Any]]:
    """The listing, in insertion order — what the app's strip and
    ``sess.checkpoints()`` show."""
    return [_meta(name, entry) for name, entry in _store.items()]


def _push() -> None:
    """Mirror a store change to open editor tabs (fire-and-forget, loop-safe —
    a no-op with no tabs/server)."""
    try:
        from .ws import manager

        manager.broadcast_threadsafe({"type": "checkpoints", "checkpoints": metas()})
    except Exception:
        pass


def save_entry(name: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Store a ready-made checkpoint dict under ``name`` (overwrites) — the
    runner's autosave path writes its rolling entry here; save() below builds
    one from the last finished run. Writes through to disk when persistence
    is enabled."""
    _store[name] = {
        "checkpoint": checkpoint,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _write_through(name, _store[name])
    _push()
    return _meta(name, _store[name])


def save(name: str, manager: Any = None) -> dict[str, Any]:
    """Store the last run's checkpoint under ``name`` (overwrites). The final
    weights are CPU-cloned at save time: the live model stays reachable
    (sess.model) and mutable (further training, notebook fine-tuning), so the
    stored entry must be a copy, not references. Raises ValueError when there
    is no trained model yet or the name is empty."""
    if manager is None:
        from .runner import run_manager as manager

    name = str(name or "").strip()
    if not name:
        raise ValueError("checkpoint name must not be empty")
    checkpoint = manager.checkpoint()  # raises ValueError without a trained model

    def clone(sd: dict[str, Any]) -> dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    checkpoint["state_dicts"] = {role: clone(sd) for role, sd in checkpoint["state_dicts"].items()}
    checkpoint["history"] = {k: list(v) for k, v in (checkpoint["history"] or {}).items()}
    return save_entry(name, checkpoint)


def load(name: str) -> dict[str, Any]:
    """The stored checkpoint dict. Unknown names raise, listing what exists.
    A hydrated placeholder materializes here — the lazy half of persistence:
    weights are read from disk on first *use*, not at session start."""
    if name not in _store:
        raise ValueError(
            f"no checkpoint named '{name}' (saved: {', '.join(sorted(_store)) or 'none'})"
        )
    entry = _store[name]
    if entry["checkpoint"] is None:
        import torch

        try:
            saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"could not read the saved checkpoint '{name}': {exc}") from None
        entry["checkpoint"] = saved["checkpoint"]
    return entry["checkpoint"]


def delete(name: str) -> None:
    if name not in _store:
        raise ValueError(
            f"no checkpoint named '{name}' (saved: {', '.join(sorted(_store)) or 'none'})"
        )
    del _store[name]
    _remove_files(name)
    _push()


def clear() -> None:
    """Empty the in-memory store (files are untouched — this is the test hook,
    not a delete-all; deleting goes entry-by-entry through delete())."""
    _store.clear()
    _push()
