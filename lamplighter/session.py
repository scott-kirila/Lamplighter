"""Lifecycle management for a Lamplighter editor session, driven from a notebook.

The server runs in a daemon thread inside the kernel and serves both the API
and the built frontend on a single port. The browser tab is just a view —
closing it loses nothing, since the backend holds the graph; reopen with
``open_editor()`` and the canvas rehydrates from the cache. The graph is also
autosaved to disk (see ``start(persist=...)``), so even a kernel restart
brings the design back.
"""
from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from . import LamplighterError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND = _PROJECT_ROOT / "frontend"
_DIST = _FRONTEND / "dist"


def _pick_port(preferred: int, wait: float = 3.0) -> int:
    """Return ``preferred``, else an OS-assigned ephemeral port.

    After a kernel restart the previous server's port lingers briefly (the old
    process is still releasing it, or it sits in TIME_WAIT), so we retry the
    preferred port for a short window before falling back. Reclaiming the same
    port lets already-open browser tabs reconnect instead of being orphaned on a
    port nothing serves anymore.
    """
    deadline = time.time() + wait
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
        if time.time() >= deadline:
            break
        time.sleep(0.2)
    # Preferred port stayed busy — fall back to an OS-assigned ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
        except OSError:
            raise LamplighterError("could not find a free port") from None


def _ensure_frontend_build(force: bool = False) -> None:
    import subprocess

    if _DIST.exists() and not force:
        return
    if not _FRONTEND.exists():
        raise LamplighterError(f"frontend directory not found at {_FRONTEND}")
    if not (_FRONTEND / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=_FRONTEND, check=True)
    subprocess.run(["npm", "run", "build"], cwd=_FRONTEND, check=True)
    if not _DIST.exists():
        raise LamplighterError("frontend build did not produce a dist/ directory")


class Session:
    """A running Lamplighter server bound to a background thread."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._server is not None
            and not self._server.should_exit
        )

    def _start_server(self) -> None:
        import uvicorn

        # Import the app only after the build exists so the static mount registers.
        from backend.app import app

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="lamplighter-uvicorn"
        )
        self._thread.start()

    def _wait_until_healthy(self, timeout: float = 25.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._server is not None and getattr(self._server, "started", False):
                try:
                    urllib.request.urlopen(f"{self.url}/api/registry", timeout=1)
                    return
                except Exception:
                    pass
            if self._thread is not None and not self._thread.is_alive():
                raise LamplighterError("server thread exited before becoming healthy")
            time.sleep(0.1)
        raise LamplighterError(f"server did not become healthy within {timeout:.0f}s")

    def open(self) -> str:
        """(Re)open the editor in the default browser. Restores work after a close."""
        webbrowser.open(self.url)
        return self.url

    def status(self) -> dict[str, Any]:
        running = self.is_running()
        has_graph = False
        if running:
            try:
                urllib.request.urlopen(f"{self.url}/api/graph", timeout=1)
                has_graph = True
            except urllib.error.HTTPError as exc:
                has_graph = exc.code != 404
            except Exception:
                pass
        return {"running": running, "url": self.url, "has_graph": has_graph}

    def stop(self) -> None:
        if self._server is not None:
            # Let open editor tabs know the session is going away before the
            # server stops, so they can reflect it instead of just retrying.
            from backend.ws import manager

            manager.notify_stopped()
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    # The session's data registry: hand the app *references* to your data —
    # nothing is copied, in-place changes are visible immediately, and
    # re-registering a name repoints it (re-run the cell after recreating data).
    # The Data tab lists exactly these, by name.
    def data(self, **objects: Any) -> dict[str, Any]:
        """Register (or repoint) named data references, e.g.
        ``sess.data(X=X, y=y)``. Calls merge, so you can add incrementally.
        With no arguments, just returns the current listing."""
        from backend import datastore

        if objects:
            try:
                datastore.register(**objects)
            except ValueError as exc:
                raise LamplighterError(str(exc)) from None
        return datastore.summary()

    def list_data(self) -> dict[str, Any]:
        """Name → metadata (kind/shape/dtype) for everything registered."""
        from backend import datastore

        return datastore.summary()

    def drop_data(self, *names: str) -> dict[str, Any]:
        """Deregister names; returns the remaining listing."""
        from backend import datastore

        try:
            datastore.drop(*names)
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None
        return datastore.summary()

    def modules(self, **classes: Any) -> list[dict[str, Any]]:
        """Register nn.Module *classes* for the Custom node, e.g.
        ``sess.modules(MyBlock=MyBlock)`` — the palette escape hatch. Calls
        merge; re-registering repoints a name (the re-run-the-cell idiom).
        With no arguments, returns the current listing. (Not to be confused
        with ``sess.models`` — the *trained* modules of the last run.)"""
        from backend import datastore

        if classes:
            try:
                datastore.register_modules(**classes)
            except ValueError as exc:
                raise LamplighterError(str(exc)) from None
        return datastore.module_summaries()

    # Bridge to runs triggered from the web app. The backend lives in this
    # kernel, so these read the run artifacts directly — no HTTP.
    @property
    def model(self):
        """The trained ``nn.Module`` from the last app-triggered run (or None)."""
        from backend.runner import run_manager

        return run_manager.model

    @property
    def models(self):
        """The trained modules from the last run, keyed by role — e.g. a GAN's
        ``{"generator": ..., "discriminator": ...}``. A single-model run exposes
        ``{"model": ...}``; ``sess.model`` is the sole module (None when there
        are several — use ``sess.models`` then)."""
        from backend.runner import run_manager

        return run_manager.models

    @property
    def history(self):
        """Per-epoch metrics dict from the last app-triggered run (or None)."""
        from backend.runner import run_manager

        return run_manager.history

    @property
    def best_model(self):
        """The model at the epoch with the lowest validation loss — often better
        than the (possibly overfit) final ``sess.model``. None when the run had
        no validation."""
        from backend.runner import run_manager

        return run_manager.best_model()

    def run_status(self) -> dict[str, Any]:
        """State of the current/last app-triggered run."""
        from backend.runner import run_manager

        return run_manager.status()

    def save_checkpoint(self, path: str = "model.pt") -> str:
        """Save the last run's trained weights + reproducibility snapshot to
        ``path`` as one self-contained file — reload it anywhere (no session or
        graph needed) with ``lamplighter.load_checkpoint(path)``."""
        import torch

        from backend.runner import run_manager

        try:
            checkpoint = run_manager.checkpoint()
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None
        torch.save(checkpoint, path)
        return path

    # The session's checkpoint store: named in-kernel snapshots of finished
    # runs, shared with the app's Checkpoints strip (saved either place, listed
    # both places). For a file on disk instead, use save_checkpoint(path).
    def checkpoint(self, name: str) -> dict[str, Any]:
        """Store the last run's checkpoint under ``name`` (overwrites) —
        restorable from the app or with ``sess.restore(name)``."""
        from backend import checkpoints

        try:
            return checkpoints.save(name)
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None

    def checkpoints(self) -> list[dict[str, Any]]:
        """Metadata for the stored checkpoints (name, created, epoch, …)."""
        from backend import checkpoints

        return checkpoints.metas()

    def restore(self, name: str) -> dict[str, Any]:
        """Repopulate the run artifacts (``sess.model``/``history``/``snapshot``)
        from a stored checkpoint, as if that run had just finished. Refused
        while a run is in progress. Returns the new run status."""
        from backend import checkpoints
        from backend.runner import run_manager

        try:
            entry = checkpoints.load(name)
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None
        error = run_manager.restore(entry)
        if error is not None:
            raise LamplighterError(error)
        return run_manager.status()

    def resume(self, name: str, epochs: int | None = None) -> dict[str, Any]:
        """Warm-start from a stored checkpoint, continuing toward its planned
        epoch target: an interrupted (or autosaved) run finishes its plan;
        ``epochs`` — a total, like everywhere else — raises the target to
        train a finished run further (e.g. ``epochs=18`` on a completed
        12-epoch run trains 6 more). Same graph/config/data picks, final
        weights loaded, fresh optimizer, new recorded seed; epoch numbering
        and the history continue where they left off."""
        from backend import checkpoints
        from backend.runner import run_manager

        try:
            entry = checkpoints.load(name)
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None
        error = run_manager.resume(name, entry, epochs=epochs)
        if error is not None:
            raise LamplighterError(error)
        return run_manager.status()

    @property
    def snapshot(self) -> dict[str, Any] | None:
        """Full reproducibility record of the current/last run: the seed,
        resolved device, effective configs, the graph, and the exact generated
        sources that ran. Replay with ``torch.manual_seed(snap["seed"])`` and
        the same sources/data."""
        from backend.runner import run_manager

        return run_manager.snapshot

    # Convenience: client calls bound to this session's URL.
    def build_model(self):
        from . import build_model as _build_model

        return _build_model(self.url)

    def model_code(self) -> str:
        from . import model_code as _model_code

        return _model_code(self.url)

    def graph(self) -> dict[str, Any]:
        from . import graph as _graph

        return _graph(self.url)

    def build_trainer(self):
        from . import build_trainer as _build_trainer

        return _build_trainer(self.url)

    def training_code(self) -> str:
        from . import training_code as _training_code

        return _training_code(self.url)

    def build_dataloaders(self):
        from . import build_dataloaders as _build_dataloaders

        return _build_dataloaders(self.url)

    def data_code(self) -> str:
        from . import data_code as _data_code

        return _data_code(self.url)

    def __repr__(self) -> str:
        state = "running" if self.is_running() else "stopped"
        return f"<Lamplighter session {self.url} ({state})>"


_current: Session | None = None


def start(
    port: int = 8000,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    build: str | bool = "auto",
    persist: bool | str = True,
) -> Session:
    """Start (or reuse) a Lamplighter session and open the editor.

    ``build`` may be ``"auto"`` (build the frontend only if missing), ``True``
    (always rebuild), or ``False`` (never build — fail if dist/ is absent).

    ``persist`` autosaves the graph on every edit and restores it when the
    backend starts empty, so a kernel restart doesn't lose the canvas. ``True``
    uses ``.lamplighter/graph.json`` in the working directory (per-project); a
    string/path picks the file; ``False`` disables it (scratch sessions).
    Saved checkpoints persist too (a ``checkpoints/`` dir beside the design
    file), so a restart keeps the runs you named — weights load lazily, only
    when an entry is used.
    """
    global _current

    if _current is not None and _current.is_running():
        if open_browser:
            _current.open()
        return _current

    if build is True:
        _ensure_frontend_build(force=True)
    elif build == "auto":
        _ensure_frontend_build(force=False)
    elif not _DIST.exists():
        raise LamplighterError(
            f"no frontend build at {_DIST} — run `npm run build` in frontend/ "
            f"or call start(build=True)"
        )

    # Graph autosave: enable the write-through and seed an empty backend from
    # the saved design — before the server accepts tabs, so the first connect
    # hydrates from it. A backend that still holds a graph wins over the file.
    # Checkpoints persist alongside (a `checkpoints/` dir next to the design
    # file): saved runs now survive a kernel restart, hydrating their listing
    # lazily — weights load only when an entry is actually used.
    from backend import checkpoints as _checkpoints
    from backend import persist as _persist

    if persist:
        design_path = Path(".lamplighter/graph.json" if persist is True else persist)
        _persist.enable(design_path)
        _checkpoints.enable(design_path.parent / "checkpoints")
    else:
        _persist.configure(None)
        _checkpoints.configure(None)

    chosen = _pick_port(port)
    session = Session(host, chosen)
    session._start_server()
    session._wait_until_healthy()
    _current = session

    if open_browser:
        session.open()
    return session


def stop() -> None:
    """Stop the current session, if any."""
    global _current
    if _current is not None:
        _current.stop()
        _current = None


def open_editor() -> str:
    """Reopen the editor browser tab for the running session."""
    if _current is None or not _current.is_running():
        raise LamplighterError("no running session — call lamplighter.start() first")
    return _current.open()


def status() -> dict[str, Any]:
    """Status of the current session."""
    if _current is None:
        return {"running": False, "url": None, "has_graph": False}
    return _current.status()


def current() -> Session | None:
    """Return the current session object, or None."""
    return _current
