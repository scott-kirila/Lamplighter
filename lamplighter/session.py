"""Lifecycle management for a Lamplighter editor session, driven from a notebook.

The server runs in a daemon thread inside the kernel and serves both the API
and the built frontend on a single port. The browser tab is just a view —
closing it loses nothing, since the backend holds the graph; reopen with
``open_editor()`` and the canvas rehydrates from the cache.
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

    # Bridge to runs triggered from the web app. The backend lives in this
    # kernel, so these read the run artifacts directly — no HTTP.
    @property
    def model(self):
        """The trained ``nn.Module`` from the last app-triggered run (or None)."""
        from backend.runner import run_manager

        return run_manager.model

    @property
    def history(self):
        """Per-epoch metrics dict from the last app-triggered run (or None)."""
        from backend.runner import run_manager

        return run_manager.history

    def run_status(self) -> dict[str, Any]:
        """State of the current/last app-triggered run."""
        from backend.runner import run_manager

        return run_manager.status()

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

    def __repr__(self) -> str:
        state = "running" if self.is_running() else "stopped"
        return f"<Lamplighter session {self.url} ({state})>"


_current: Session | None = None


def start(
    port: int = 8000,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    build: str | bool = "auto",
) -> Session:
    """Start (or reuse) a Lamplighter session and open the editor.

    ``build`` may be ``"auto"`` (build the frontend only if missing), ``True``
    (always rebuild), or ``False`` (never build — fail if dist/ is absent).
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
