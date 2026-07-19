"""Locate the built frontend — the Vite bundle the backend serves.

Two homes, checked in order:

- **Packaged**: release wheels bundle the built UI inside the package at
  ``lamplighter/_frontend_dist`` (see pyproject's ``force-include``), so an
  installed library needs no Node toolchain at all.
- **Dev checkout**: the repo's ``frontend/dist``, produced by ``npm run build``
  (``lamplighter.session`` builds it on demand).

Plain ``Path(__file__)`` resolution rather than ``importlib.resources``:
FastAPI's ``StaticFiles`` needs a real on-disk directory either way, and
wheels install unpacked — the two are equivalent here, and this reads plainer.
"""
from pathlib import Path

# lamplighter/backend/dist.py → parent.parent = the lamplighter package dir.
_PACKAGED = Path(__file__).resolve().parent.parent / "_frontend_dist"
# …and one more up = the repo root in a source checkout (site-packages when
# installed, where frontend/ simply doesn't exist).
_REPO_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def frontend_dist() -> Path | None:
    """The directory holding the built UI, or None when neither home has one
    (a source checkout before its first ``npm run build``)."""
    if (_PACKAGED / "index.html").exists():
        return _PACKAGED
    dev = _REPO_FRONTEND / "dist"
    if (dev / "index.html").exists():
        return dev
    return None


def repo_frontend_dir() -> Path | None:
    """The checkout's ``frontend/`` source dir (for dev builds), or None when
    this is an installed package rather than a checkout."""
    return _REPO_FRONTEND if (_REPO_FRONTEND / "package.json").exists() else None
