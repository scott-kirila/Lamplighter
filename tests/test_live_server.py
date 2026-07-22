"""The real server, over a real socket.

Every other HTTP/WS test drives the ASGI app in-process through Starlette's
``TestClient``, which never opens a port, never runs uvicorn, and never
exercises the ``websockets`` library the browser handshake actually uses. So the
whole boot path — pick a port, start the thread, wait for health, serve the
bundled UI, accept a real WebSocket — had no coverage at all, in the one file
(``session.py``) a user's very first line of code runs.

These are deliberately few and slow-ish: enough to prove the process starts and
answers, not a second copy of the API suite.
"""
import json
import socket

import pytest

import lamplighter
from lamplighter.backend import state
from lamplighter.backend.templates import TEMPLATES


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live():
    """One real session for the module — booting is the expensive part.

    Restores every global it touches. A real session sets both the cached
    project and the module-level `_current`, and leaving either behind is not a
    local mess: the next file to read `state.get_project()` sees this module's
    template, and the next `Lamplighter()` adopts a stopped server.
    """
    from lamplighter import session as session_mod

    prior_project = state.get_project()
    prior_current = session_mod._current

    server = lamplighter.Lamplighter(port=_free_port(), persist=False, build=False)
    state.set_project(TEMPLATES["mnist"].build())
    try:
        yield server
    finally:
        server.stop()
        state._current = prior_project
        session_mod._current = prior_current


def test_the_server_actually_boots_and_answers(live):
    import urllib.request

    assert live.is_running()
    with urllib.request.urlopen(f"{live.url}/api/registry", timeout=10) as resp:
        assert resp.status == 200
        assert json.load(resp)  # a real registry, not an empty body


def test_it_serves_the_bundled_ui(live):
    """The static mount is registered at import time from whatever
    frontend_dist() resolves — in a wheel that is the packaged copy, and CI's
    backend job has historically skipped it entirely."""
    import urllib.request

    from lamplighter.backend.dist import frontend_dist

    if frontend_dist() is None:
        pytest.skip("no built frontend in this checkout")
    with urllib.request.urlopen(live.url, timeout=10) as resp:
        body = resp.read().decode()
    assert resp.status == 200
    assert "<div id=\"root\">" in body or "<script" in body
    # The upgrade guard: a cached index.html pointing at a deleted bundle is a
    # blank page (see _HashAwareStatics).
    assert resp.headers.get("cache-control") == "no-cache"


@pytest.mark.parametrize(
    "origin,accepted",
    [
        ("http://127.0.0.1:9999", True),   # the editor tab, tunnelled or not
        ("http://localhost:9999", True),
        (None, True),                      # the notebook client / curl
        ("http://evil.com", False),        # a page the user merely visited
        ("null", False),                   # sandboxed iframe / file://
    ],
)
def test_the_origin_boundary_holds_over_a_real_handshake(live, origin, accepted):
    """TestClient's in-process socket can't prove this: it never performs the
    HTTP upgrade that carries Origin, and never runs the server's rejection
    through a real client."""
    import asyncio

    websockets = pytest.importorskip("websockets")

    async def connect():
        kwargs = {"origin": origin} if origin is not None else {}
        async with websockets.connect(f"ws://127.0.0.1:{live.port}/ws", **kwargs) as ws:
            return json.loads(await asyncio.wait_for(ws.recv(), 10))

    if accepted:
        msg = asyncio.run(connect())
        assert msg["type"] == "sync"
        assert "model" in msg["models"]
    else:
        with pytest.raises(Exception):
            asyncio.run(connect())


def test_a_second_session_adopts_the_first(live):
    """Re-running the setup cell is the most common thing a notebook user does;
    it must not start a second server or orphan the open tab."""
    again = lamplighter.Lamplighter(port=_free_port(), persist=False, build=False)
    assert again is live
    assert again.port == live.port
