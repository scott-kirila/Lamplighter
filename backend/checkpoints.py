"""The session's named checkpoint store.

Saving snapshots the last run's checkpoint (the same self-contained format the
weights download and ``sess.save_checkpoint()`` use) into kernel memory under a
name — no files involved. Entries can be listed, downloaded, deleted, and
restored (repopulating the run manager as if that run had just finished), and
they're what warm-start resume trains from.

Kernel-side and server-side code share this module in-process (same pattern as
the data registry). A kernel restart clears it, like any kernel object. Every
mutation pushes the fresh listing to open editor tabs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

_store: dict[str, dict[str, Any]] = {}


def _meta(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """A listing row: identity + the numbers that distinguish checkpoints
    (where training stood, how good it was, how to reproduce it)."""
    checkpoint = entry["checkpoint"]
    val = (checkpoint.get("history") or {}).get("val_loss") or []
    return {
        "name": name,
        "created": entry["created"],
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "seed": (checkpoint.get("snapshot") or {}).get("seed"),
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
    one from the last finished run."""
    _store[name] = {
        "checkpoint": checkpoint,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
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
    checkpoint["state_dict"] = {
        k: v.detach().cpu().clone() for k, v in checkpoint["state_dict"].items()
    }
    checkpoint["history"] = {k: list(v) for k, v in (checkpoint["history"] or {}).items()}
    return save_entry(name, checkpoint)


def load(name: str) -> dict[str, Any]:
    """The stored checkpoint dict. Unknown names raise, listing what exists."""
    if name not in _store:
        raise ValueError(
            f"no checkpoint named '{name}' (saved: {', '.join(sorted(_store)) or 'none'})"
        )
    return _store[name]["checkpoint"]


def delete(name: str) -> None:
    if name not in _store:
        raise ValueError(
            f"no checkpoint named '{name}' (saved: {', '.join(sorted(_store)) or 'none'})"
        )
    del _store[name]
    _push()


def clear() -> None:
    _store.clear()
    _push()
