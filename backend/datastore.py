"""The session's named data registry.

The notebook hands the session *references* to its data objects —
``sess.data(X=X, y=y)`` — and the app's Data tab lists exactly those, instead of
scanning the whole notebook namespace (which doesn't scale past a handful of
variables). Nothing is copied: entries are references, so in-place mutation is
visible immediately, and re-registering a name repoints it (the natural
"re-run the cell" idiom). The app links to *names*; the runner resolves
name → object at run start, so data can change without re-picking in the app.

Kernel-side and server-side code share this module in-process (same pattern as
the run manager). A kernel restart clears it, like any kernel object.
"""
from __future__ import annotations

from typing import Any

from .introspect import _describe

_registry: dict[str, Any] = {}


def register(**objects: Any) -> None:
    """Register (or repoint) named data references. Calls merge, so registering
    incrementally across cells works. Rejects non-data objects up front with a
    clear message rather than letting them vanish from the picker silently."""
    for name, value in objects.items():
        if _describe(name, value) is None:
            raise ValueError(
                f"'{name}' is a {type(value).__name__}, not a data object — expected a "
                "torch.Tensor, numpy array, Dataset, or DataLoader"
            )
    _registry.update(objects)


def drop(*names: str) -> None:
    """Deregister names. Unknown names raise, listing what is registered."""
    unknown = [n for n in names if n not in _registry]
    if unknown:
        raise ValueError(
            f"not registered: {', '.join(unknown)} (registered: {', '.join(sorted(_registry)) or 'nothing'})"
        )
    for name in names:
        del _registry[name]


def clear() -> None:
    _registry.clear()


def registry() -> dict[str, Any]:
    """The live name → object mapping (references, not copies)."""
    return _registry


def summary() -> dict[str, dict[str, Any]]:
    """Name → metadata (kind/shape/dtype/…) for listing in the notebook."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in _registry.items():
        d = _describe(name, value) or {"name": name, "kind": "unknown"}
        d.pop("name", None)
        out[name] = d
    return out
