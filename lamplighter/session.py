"""Lifecycle management for a Lamplighter editor session, driven from a notebook.

The server runs in a daemon thread inside the kernel and serves both the API
and the built frontend on a single port. The browser tab is just a view —
closing it loses nothing, since the backend holds the graph; reopen with
``sess.open()`` and the canvas rehydrates from the cache. The graph is also
autosaved to disk (see ``Lamplighter(persist=...)``), so even a kernel
restart brings the design back.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import urllib.error
import urllib.request
import warnings
import webbrowser
from pathlib import Path
from typing import Any

from . import LamplighterError
from .backend.dist import frontend_dist, repo_frontend_dir

# Hosts that keep the server reachable only from this machine. Anything else
# exposes it — see _warn_if_exposed.
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _warn_if_exposed(host: str) -> None:
    """The server has NO authentication — it's designed to sit on localhost
    beside the kernel (remote use goes through an SSH tunnel; see open()).
    Binding anything else means whoever can reach the port can drive this
    kernel: start training runs, read the registered data listing, download
    trained weights. Say so loudly rather than silently serving."""
    if host not in _LOCAL_HOSTS:
        warnings.warn(
            f"Lamplighter is binding to {host!r} with no authentication — anyone who "
            "can reach this port can drive this kernel (start runs, read registered "
            "data, download weights). Prefer the default 127.0.0.1 and an SSH tunnel "
            "(sess.open() prints the command on remote kernels).",
            stacklevel=3,
        )

def _likely_remote() -> bool:
    """Does this kernel look like it's not on the user's machine (an SSH
    session, or a display-less Linux box)? There, ``webbrowser`` would open a
    browser on the *server* (or fail into a console one) — useless either way,
    so ``open()`` prints how to reach the app instead. A heuristic default
    only: ``Lamplighter(remote=True/False)`` states the truth and silences it,
    and an explicit ``BROWSER`` env var counts as knowing what you're doing."""
    if os.environ.get("BROWSER"):
        return False
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    import sys

    return (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )


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
    """Make sure a built UI exists somewhere frontend_dist() can find it.

    Installed wheels bundle the UI, so this returns immediately there. The npm
    path is DEV-ONLY: it applies in a source checkout (frontend/ sources
    present), building the Vite bundle on first use. Anything else — no bundle
    and no checkout — is a broken install, said plainly."""
    import subprocess

    if not force and frontend_dist() is not None:
        return
    frontend = repo_frontend_dir()
    if frontend is None:
        raise LamplighterError(
            "no frontend build found and no frontend/ sources to build one from — "
            "installed wheels ship the UI prebuilt, so this looks like a broken "
            "install (or a source checkout missing frontend/); try reinstalling"
        )
    try:
        if not (frontend / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=frontend, check=True)
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)
    except FileNotFoundError:
        # npm itself is absent — only dev checkouts ever need it.
        raise LamplighterError(
            "building the frontend needs Node.js/npm (a dev-checkout-only step — "
            "release wheels ship the UI prebuilt); install it from nodejs.org, "
            "or run: cd frontend && npm install && npm run build"
        ) from None
    if frontend_dist() is None:
        raise LamplighterError("frontend build did not produce a dist/ directory")


class Session:
    """A running Lamplighter server bound to a background thread."""

    def __init__(self, host: str, port: int, remote: bool | None = None) -> None:
        self.host = host
        self.port = port
        # Is this kernel somewhere the user's browser isn't? None = auto-detect.
        self.remote = remote
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
        from .backend.app import app
        from .backend import origins

        # Loopback is always allowed; this only matters for the Lamplighter(host=…)
        # case, which _warn_if_exposed has already objected to. Without it the
        # Host check below would refuse the very interface the user chose to bind.
        origins.allow_host(self.host)

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

    def open(self, tab: str | None = None) -> str:
        """Open the editor in the default browser — the one way a tab opens.

        ``tab="training"`` lands on the Training tab instead of the canvas;
        :func:`demo` uses it so a first run is one click rather than a hunt.

        On a remote kernel (``self.remote``, auto-detected when None) a browser
        opened *here* wouldn't reach you, so instead print how to reach the app
        from your machine."""
        url = f"{self.url}?tab={tab}" if tab else self.url
        remote = self.remote if self.remote is not None else _likely_remote()
        if remote:
            print(
                f"this session looks remote — a browser opened here wouldn't reach you.\n"
                f"From your machine, e.g.:\n"
                f"  ssh -L {self.port}:127.0.0.1:{self.port} <this-host>\n"
                f"then open {url}\n"
                f"(pass remote=False to Lamplighter() if this detection is wrong)"
            )
        else:
            webbrowser.open(url)
        return url

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
            from .backend.ws import manager

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
        from .backend import datastore

        if objects:
            try:
                datastore.register(**objects)
            except ValueError as exc:
                raise LamplighterError(str(exc)) from None
        return datastore.summary()

    def list_data(self) -> dict[str, Any]:
        """Name → metadata (kind/shape/dtype) for everything registered."""
        from .backend import datastore

        return datastore.summary()

    def drop_data(self, *names: str) -> dict[str, Any]:
        """Deregister names; returns the remaining listing."""
        from .backend import datastore

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
        from .backend import datastore

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
        from .backend.runner import run_manager

        return run_manager.model

    @property
    def models(self):
        """The trained modules from the last run, keyed by role — e.g. a GAN's
        ``{"generator": ..., "discriminator": ...}``. A single-model run exposes
        ``{"model": ...}``; ``sess.model`` is the sole module (None when there
        are several — use ``sess.models`` then)."""
        from .backend.runner import run_manager

        return run_manager.models

    @property
    def history(self):
        """Per-epoch metrics dict from the last app-triggered run (or None)."""
        from .backend.runner import run_manager

        return run_manager.history

    @property
    def best_model(self):
        """The model at the epoch with the lowest validation loss — often better
        than the (possibly overfit) final ``sess.model``. None when the run had
        no validation."""
        from .backend.runner import run_manager

        return run_manager.best_model()

    def run_status(self) -> dict[str, Any]:
        """State of the current/last app-triggered run."""
        from .backend.runner import run_manager

        return run_manager.status()

    def save_checkpoint(self, path: str = "model.pt") -> str:
        """Save the last run's trained weights + reproducibility snapshot to
        ``path`` as one self-contained file — reload it anywhere (no session or
        graph needed) with ``lamplighter.load_checkpoint(path)``."""
        import torch

        from .backend.runner import run_manager

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
        from .backend import checkpoints

        try:
            return checkpoints.save(name)
        except ValueError as exc:
            raise LamplighterError(str(exc)) from None

    def checkpoints(self) -> list[dict[str, Any]]:
        """Metadata for the stored checkpoints (name, created, epoch, …)."""
        from .backend import checkpoints

        return checkpoints.metas()

    def restore(self, name: str) -> dict[str, Any]:
        """Repopulate the run artifacts (``sess.model``/``history``/``snapshot``)
        from a stored checkpoint, as if that run had just finished. Refused
        while a run is in progress. Returns the new run status."""
        from .backend import checkpoints
        from .backend.runner import run_manager

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
        from .backend import checkpoints
        from .backend.runner import run_manager

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
        from .backend.runner import run_manager

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


class Lamplighter(Session):
    """The session manager: constructing it brings the session up (a server
    thread in this kernel) *without* opening a browser tab — attach data with
    ``.data(X=X, y=y)``, then call ``.open()`` when you want the editor.

    One kernel runs one session: constructing while one is already live adopts
    it instead of booting a second server, so re-running a setup cell is
    idempotent. The adopted session keeps its original port/build/persistence —
    arguments on the re-run are ignored.

    ``remote`` says whether this kernel is somewhere your browser isn't (an
    SSH'd server, say): there ``.open()`` prints how to reach the app instead
    of opening a browser on the wrong machine. The default ``None``
    auto-detects (SSH session, or display-less Linux); pass ``True``/``False``
    when you know your setup.

    ``build`` may be ``"auto"`` (build the frontend only if missing), ``True``
    (always rebuild), or ``False`` (never build — fail if dist/ is absent).

    ``persist`` autosaves the project on every edit and restores it when the
    backend starts empty, so a kernel restart doesn't lose the canvas. ``True``
    uses ``.lamplighter/graph.json`` in the working directory (per-project); a
    string/path picks the file; ``False`` disables it (scratch sessions).
    Saved checkpoints persist too (a ``checkpoints/`` dir beside the autosave
    file), so a restart keeps the runs you named — weights load lazily, only
    when an entry is used.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Lamplighter":
        if _current is not None and _current.is_running():
            return _current  # type: ignore[return-value]  # adopt the live session
        return super().__new__(cls)

    def __init__(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        remote: bool | None = None,
        build: str | bool = "auto",
        persist: bool | str = True,
    ) -> None:
        global _current
        if self is _current and self.is_running():
            return  # adopted a live session — leave it untouched

        _warn_if_exposed(host)
        if build is True:
            _ensure_frontend_build(force=True)
        elif build == "auto":
            _ensure_frontend_build(force=False)
        elif frontend_dist() is None:
            raise LamplighterError(
                "no frontend build found — run `npm run build` in frontend/ "
                "or pass build=True (installed wheels ship the UI prebuilt)"
            )

        # Graph autosave: enable the write-through and seed an empty backend from
        # the saved design — before the server accepts tabs, so the first connect
        # hydrates from it. A backend that still holds a graph wins over the file.
        # Checkpoints persist alongside (a `checkpoints/` dir next to the design
        # file): saved runs now survive a kernel restart, hydrating their listing
        # lazily — weights load only when an entry is actually used.
        from .backend import checkpoints as _checkpoints
        from .backend import persist as _persist

        if persist:
            design_path = Path(".lamplighter/graph.json" if persist is True else persist)
            _persist.enable(design_path)
            _checkpoints.enable(design_path.parent / "checkpoints")
        else:
            _persist.configure(None)
            _checkpoints.configure(None)

        super().__init__(host, _pick_port(port), remote=remote)
        self._start_server()
        self._wait_until_healthy()
        _current = self


def stop() -> None:
    """Stop the current session, if any."""
    global _current
    if _current is not None:
        _current.stop()
        _current = None


def status() -> dict[str, Any]:
    """Status of the current session."""
    if _current is None:
        return {"running": False, "url": None, "has_graph": False}
    return _current.status()


def current() -> Session | None:
    """Return the current session object, or None."""
    return _current


def demo(*, template: str = "mnist", open_browser: bool = True, **kwargs: Any) -> Lamplighter:
    """One cell from install to a loss curve.

    Starts a session, loads a template that brings its own data, and opens the
    browser on the Training tab with the Run button armed::

        import lamplighter
        sess = lamplighter.demo()     # then press ▶ Run

    Every other template is deliberately a blank cheque — it exists to be
    pointed at *your* tensors — which means a fresh install has no path from an
    empty canvas to a trained model, and that is the first thing anyone tries.
    ``mnist`` draws from torchvision instead, so there is nothing to register
    first. It is ~15s of laptop CPU for three epochs at about 98% accuracy.

    Scratch by default: ``persist=False`` so a demo can't overwrite the project
    you were working on. Pass ``persist=True`` (or anything else
    :class:`Lamplighter` accepts) to change that.
    """
    from .backend import state
    from .backend.templates import TEMPLATES

    if template not in TEMPLATES:
        raise LamplighterError(
            f"no template {template!r} — try one of: {', '.join(TEMPLATES)}"
        )

    kwargs.setdefault("persist", False)
    session = Lamplighter(**kwargs)
    state.set_project(TEMPLATES[template].build())
    if open_browser:
        session.open(tab="training")
    return session
