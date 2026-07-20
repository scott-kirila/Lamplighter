"""The session's run store (né checkpoint store).

Every terminal run auto-records here as a WEIGHTLESS entry — curves, health,
steps, and the reproducibility snapshot, under an assigned ``run-N`` name — so
run history exists without a naming ritual. "Keeping weights" (the explicit
save) upgrades an entry with CPU-cloned state dicts: **a checkpoint is just a
run that kept its weights**. Entries can be listed, viewed, renamed,
downloaded, deleted, restored (repopulating the run manager as if that run had
just finished), and weighted ones are what warm-start resume trains from.

Retention: only the newest ``_AUTO_KEEP`` weightless auto records are kept
(failed ones prune first); renaming a run or keeping its weights exempts it.

Optionally persistent (``lamplighter.start(persist=...)`` enables it alongside
the project autosave): each entry writes through to
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
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# The checkpoint FORMAT version, stamped into every checkpoint dict the runner
# builds (so stored entries, sidecars' .pt payloads, and downloaded files all
# carry it) — the hook future migrations key on. Entries without it predate
# versioning (implicitly v0). The project's OTHER version numbers, for
# orientation: schema.Project.version (the project wire/persistence shape,
# currently 3) and persist.py's autosave file wrapper (currently 2). Each
# guards a different format; this one is the checkpoint payload's.
CHECKPOINT_VERSION = 1

# name → {"checkpoint": dict | None, "created": str, ["auto", "meta", "path"]}.
# checkpoint is None for a hydrated-but-not-yet-loaded entry (its listing row
# rides "meta"; load() materializes from "path" on first use). "auto" marks an
# auto-recorded run still subject to retention pruning.
_store: dict[str, dict[str, Any]] = {}
_dir: Path | None = None

# Guards _store's structure: the training thread mutates it (record/autosave/
# prune) while FastAPI's worker threads list/rename/delete concurrently — an
# unguarded metas() iteration during an insert raises "dict changed size", and
# rename's rebuild would expose an empty store mid-swap. Heavy file I/O
# (torch.save/load) deliberately stays OUTSIDE the lock so persisting a big
# autosave never blocks a listing poll. RLock: rename materializes via load()
# under its own acquisition.
_lock = threading.RLock()

# Auto-recorded (weightless, never renamed) runs retained; see _prune().
_AUTO_KEEP = 25


def configure(directory: Path | str | None) -> None:
    """Point the write-through at a directory; None disables it (the default)."""
    global _dir
    _dir = Path(directory) if directory is not None else None


def enable(directory: Path | str) -> None:
    """The session-start hook: configure the write-through AND hydrate the
    listing from the meta sidecars — lazily (no weights are read here). An
    entry already in the live store wins over its file (it is at least as
    fresh, same rule as the project autosave)."""
    configure(directory)
    assert _dir is not None
    if not _dir.exists():
        return
    # Collect valid rows first, then insert in CHRONOLOGICAL order: insertion
    # order IS the listing's chronology (the frontend shows newest-first by
    # reversing it, and "latest run" reads the tail), but the files sort
    # lexicographically — run-10 before run-2 — so hydrate by each sidecar's
    # own created stamp instead.
    rows: list[tuple[str, str, dict, Path]] = []  # (created, name, meta, pt)
    for sidecar in sorted(_dir.glob("*.json")):
        if sidecar.name == "run-counter.json":
            continue  # the run-name counter, not a checkpoint sidecar
        try:
            meta = json.loads(sidecar.read_text())
            name = str(meta["name"])
            pt = sidecar.with_suffix(".pt")
            if name in _store or not pt.exists():
                continue
            # A current sidecar carries the full listing row; a stale/incomplete
            # one is missing these keys, so the KeyError skips it (start blank for
            # that entry) rather than hydrating a half row.
            _ = (meta["created"], meta["state"], meta["has_weights"], meta["auto"])
            rows.append((str(meta["created"]), name, meta, pt))
        except Exception as exc:
            warnings.warn(f"ignoring the saved checkpoint at {sidecar} ({exc})", stacklevel=2)
    with _lock:
        for created, name, meta, pt in sorted(rows, key=lambda r: (r[0], r[1])):
            _store[name] = {
                "checkpoint": None,
                "created": created,
                # Keep auto-ness on the entry too: load() materializes the
                # checkpoint and _meta then reads the entry, not the sidecar.
                "auto": bool(meta["auto"]),
                "meta": meta,
                "path": pt,
            }
    if rows:
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


def run_models_from(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Which model(s) a run trained, from its snapshot — one ``{role, id, name}``
    per trained model, with the NAME frozen at run time (read from the snapshot's
    own project dump, so a later rename/delete never blanks an existing row).
    Mirrors the runner's role assignment (``_assign_roles``): the explicit
    ``training.roles`` when present, else the sole model for an auto-assigned
    single-model run. Kept here (not in runner) so the listing can attribute any
    stored run without importing the runner — same reason ``_meta`` lives here."""
    if not snapshot:
        return []
    models = (snapshot.get("project") or {}).get("models") or []
    by_id = {m.get("id"): m.get("name") for m in models}
    roles = (snapshot.get("training") or {}).get("roles") or {}
    if roles:
        return [
            {"role": role, "id": mid, "name": by_id.get(mid)}
            for role, mid in roles.items()
            if mid is not None
        ]
    # Auto-assigned single-model run (roles empty): the sole model is the target.
    if len(models) == 1:
        m = models[0]
        return [{"role": "model", "id": m.get("id"), "name": m.get("name")}]
    return []


def _meta(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """A listing row: identity + the numbers that distinguish runs (where
    training stood, how good it was, how to reproduce it, whether weights were
    kept). A hydrated placeholder answers from its sidecar meta — no weights
    are read to list."""
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
        "state": snapshot.get("state"),
        "source": snapshot.get("source", "app"),
        # The run's recipe name — lets the Runs list label an RL run's progress
        # in iterations rather than epochs. None for pre-recipe sidecars.
        "recipe": (snapshot.get("training") or {}).get("recipe"),
        "has_weights": checkpoint.get("state_dicts") is not None,
        "auto": bool(entry.get("auto", False)),
        # Which model(s) this run trained — role → {id, name} (name frozen at run
        # time). Lets the Runs list scope/label by model; [] for old snapshots.
        "models": run_models_from(snapshot),
        # The sweep study this run belongs to (an Optimize trial), or None —
        # the trials table is DERIVED by filtering the listing on this.
        "study": snapshot.get("study"),
    }


def _is_auto(entry: dict[str, Any]) -> bool:
    if entry["checkpoint"] is None:
        return bool(entry["meta"]["auto"])
    return bool(entry.get("auto", False))


def _has_weights(entry: dict[str, Any]) -> bool:
    if entry["checkpoint"] is None:
        return bool(entry["meta"]["has_weights"])
    return entry["checkpoint"].get("state_dicts") is not None


def is_auto(name: str) -> bool:
    """True when ``name`` is an existing auto record — a reserved run-N slot
    that belongs to one specific run's curves. Keep-weights must not overwrite
    such a slot with a different run's live model (it would mislabel them);
    False for absent names and user-named saves, which are free to overwrite."""
    with _lock:
        entry = _store.get(name)
    return entry is not None and _is_auto(entry)


def metas() -> list[dict[str, Any]]:
    """The listing, in insertion order — what the app's strip and
    ``sess.checkpoints()`` show. Snapshot the items under the lock so a
    concurrent record/prune (training thread) can't change the dict mid-walk."""
    with _lock:
        items = list(_store.items())
    return [_meta(name, entry) for name, entry in items]


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
    entry = {
        "checkpoint": checkpoint,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        _store[name] = entry
    _write_through(name, entry)
    _push()
    return _meta(name, entry)


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


def strip_weights(name: str) -> None:
    """Demote an entry back to a weightless auto record — the sweep engine's
    dethroned-best cleanup: an interim best kept its weights (and left the
    retention pool) while it led; once beaten, its weights go and retention
    applies again, so a long sweep doesn't accrete permanent interim rows.
    Silently a no-op for unknown names (renamed/deleted under our feet)."""
    with _lock:
        entry = _store.get(name)
    if entry is None:
        return
    if entry["checkpoint"] is None:
        try:
            load(name)  # materialize so the rewrite drops the weights on disk too
        except ValueError:
            return
    entry["checkpoint"]["state_dicts"] = None
    entry["checkpoint"]["best_state_dict"] = None
    entry["auto"] = True
    _write_through(name, entry)
    _push()


def next_run_name() -> str:
    """The next auto-record name (``run-N``). The counter persists beside the
    entries so renames can never cause a number's reuse; without persistence
    (or a fresh dir) it derives from the names in the store."""
    n = 0
    counter = _dir / "run-counter.json" if _dir is not None else None
    if counter is not None:
        try:
            n = int(json.loads(counter.read_text())["next"])
        except Exception:
            n = 0
    with _lock:
        names = list(_store)
    for name in names:
        m = re.fullmatch(r"run-(\d+)", name)
        if m:
            n = max(n, int(m.group(1)))
    n += 1
    if counter is not None:
        try:
            _dir.mkdir(parents=True, exist_ok=True)
            tmp = counter.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"next": n}))
            os.replace(tmp, counter)
        except Exception:
            pass
    return f"run-{n}"


def record(manager: Any) -> dict[str, Any] | None:
    """Auto-record a terminal run as a WEIGHTLESS entry under the name the
    run reserved at start — every run joins the history without being asked.
    Failed runs record too (a sweep's failures are data). Never raises into
    the run thread; retention prunes afterwards."""
    try:
        rec = manager.run_record()
        name = getattr(manager, "run_name", None) or next_run_name()
        entry = {
            "checkpoint": rec,
            "created": datetime.now().isoformat(timespec="seconds"),
            "auto": True,
        }
        with _lock:
            _store[name] = entry
        _write_through(name, entry)
        _prune()
        _push()
        return _meta(name, entry)
    except Exception as exc:
        warnings.warn(f"could not record the run: {exc}", stacklevel=2)
        return None


def _prune() -> None:
    """Bound the auto history: keep the newest ``_AUTO_KEEP`` weightless auto
    entries — in TWO independent pools, sweep trials (a ``study`` tag) and
    regular runs, so a 20-trial sweep can't evict your training history (nor
    the reverse). Renaming a run or keeping its weights exempts it; within
    each pool, failed runs prune before finished ones (they're data, but the
    cheapest kind). The select+delete happens under the lock; the file
    removals after it."""
    with _lock:
        autos = [(name, e) for name, e in _store.items() if _is_auto(e) and not _has_weights(e)]

        def prune_order(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
            name, e = item
            failed = _meta(name, e).get("state") == "failed"
            return (0 if failed else 1, e.get("created", ""))

        victims: list[str] = []
        for is_trial in (True, False):
            pool = [it for it in autos if bool(_meta(it[0], it[1]).get("study")) == is_trial]
            if len(pool) <= _AUTO_KEEP:
                continue
            victims += [name for name, _ in sorted(pool, key=prune_order)[: len(pool) - _AUTO_KEEP]]
        for name in victims:
            del _store[name]
    for name in victims:
        _remove_files(name)


def rename(old: str, new: str) -> dict[str, Any]:
    """Rename an entry, in place in the listing order. Naming a run is keep
    intent — the auto flag clears, exempting it from retention pruning."""
    new = str(new or "").strip()
    if not new:
        raise ValueError("run name must not be empty")
    with _lock:
        if old not in _store:
            raise ValueError(f"no run named '{old}' (saved: {', '.join(sorted(_store)) or 'none'})")
        if new != old and new in _store:
            raise ValueError(f"a run named '{new}' already exists")

    load(old)  # materialize so the files can be rewritten under the new stem
    with _lock:
        entry = _store[old]
        entry["auto"] = False
        if new != old:
            # Rebuild in place under the lock, so a concurrent listing never
            # observes the store mid-swap (empty, or with both names).
            renamed = {(new if k == old else k): v for k, v in _store.items()}
            _store.clear()
            _store.update(renamed)
    if new != old:
        _remove_files(old)
    _write_through(new, entry)
    _push()
    return _meta(new, entry)


def load(name: str) -> dict[str, Any]:
    """The stored checkpoint dict. Unknown names raise, listing what exists.
    A hydrated placeholder materializes here — the lazy half of persistence:
    weights are read from disk on first *use*, not at session start."""
    with _lock:
        if name not in _store:
            raise ValueError(
                f"no checkpoint named '{name}' (saved: {', '.join(sorted(_store)) or 'none'})"
            )
        entry = _store[name]
    if entry["checkpoint"] is None:
        import torch

        # The torch.load stays outside the lock (it can be a big weights read);
        # two racing loads both read the same file — last assignment wins with
        # an identical value.
        try:
            saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"could not read the saved checkpoint '{name}': {exc}") from None
        entry["checkpoint"] = saved["checkpoint"]
    return entry["checkpoint"]


def delete(name: str) -> None:
    with _lock:
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
    with _lock:
        _store.clear()
    _push()
