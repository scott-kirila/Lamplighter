"""The localhost trust boundary.

These are adversarial: each test is the attack the check exists to stop, written
from the attacker's side. A browser sends these headers on the user's behalf, so
"only this machine can reach the port" is not by itself the boundary the README
claims — see backend/origins.py.
"""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.helpers import edge, graph, node, single_model_project
from lamplighter.backend import origins, persist, state
from lamplighter.backend.app import app


@pytest.fixture(autouse=True)
def _isolated():
    prior = state.get_project()
    state._current = None
    persist.configure(None)
    yield
    persist.configure(None)
    state._current = prior


@pytest.fixture
def client():
    # A cached project is what makes the connect-time `sync` fire — which is
    # exactly the payload a foreign origin must never receive, so every socket
    # test needs one to be meaningful.
    state.set_project(
        single_model_project(
            graph(
                [node("in", "Input", {"shape": "1, 8"}),
                 node("l", "Linear", {"out_features": 3}),
                 node("out", "Output")],
                [edge("in", "l"), edge("l", "out")],
            )
        )
    )
    with TestClient(app) as c:
        yield c


# --- the header parser (fails closed) ---------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://127.0.0.1:8000", "127.0.0.1"),
        ("http://localhost:8000", "localhost"),
        ("https://LocalHost", "localhost"),        # scheme and case are irrelevant
        ("127.0.0.1:8000", "127.0.0.1"),           # a Host header carries no scheme
        ("http://[::1]:8000", "::1"),              # IPv6 brackets are stripped
        ("[::1]:8000", "::1"),
        ("http://evil.com", "evil.com"),
        ("null", "null"),                          # sandboxed iframe / file://
        ("", ""),
        (None, ""),
    ],
)
def test_hostname_extraction(value, expected):
    assert origins._hostname(value) == expected


# --- WebSocket: the write-capable hole --------------------------------------

@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.com",
        "https://evil.com",
        "http://127.0.0.1.evil.com",   # loopback as a prefix, not the host
        "http://localhost.evil.com",
        "null",                        # sandboxed iframe or a file:// page
    ],
)
def test_ws_rejects_foreign_origin(client, origin):
    """A page the user merely *visits* must not be able to open the socket that
    hands over the whole project and can overwrite it on disk. The refusal has
    to land before `accept()`, so nothing is ever sent."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": origin}) as ws:
            ws.receive_json()


def test_ws_rejection_cannot_write_the_project(client):
    """The write path is the real damage: `validate` replaces the project and
    persists it. A refused socket must never reach that handler."""
    before = state.get_project().model_dump()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.com"}) as ws:
            ws.send_json({"type": "validate", "project": {"models": [], "training": {}}})
            ws.receive_json()
    assert state.get_project().model_dump() == before


@pytest.mark.parametrize(
    "origin",
    ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"],
)
def test_ws_accepts_loopback_origin(client, origin):
    """The real editor tab, under every spelling loopback takes — including an
    SSH tunnel, whose local port legitimately differs from the server's."""
    with client.websocket_connect("/ws", headers={"origin": origin}) as ws:
        assert ws.receive_json()["type"] == "sync"


def test_ws_accepts_missing_origin(client):
    """Non-browser clients (the notebook client, curl, websockets) send no
    Origin — and they are not the threat this check exists for."""
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "sync"


# --- HTTP: DNS rebinding ----------------------------------------------------

@pytest.mark.parametrize("host", ["evil.com", "attacker.example:8000", "127.0.0.1.evil.com"])
def test_http_rejects_foreign_host(client, host):
    """A rebound domain resolves to 127.0.0.1 but still names itself in Host."""
    for method, path in [
        ("get", "/api/registry"),
        ("get", "/api/project"),
        ("get", "/api/run/weights"),
        ("post", "/api/run/stop"),
    ]:
        resp = getattr(client, method)(path, headers={"host": host})
        assert resp.status_code == 421, f"{method.upper()} {path} answered a foreign Host"


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000"])
def test_http_accepts_loopback_host(client, host):
    assert client.get("/api/registry", headers={"host": host}).status_code == 200


def test_allow_host_opens_a_chosen_bind(client):
    """Lamplighter(host=…) must still work — it warns, but it is the user's call."""
    assert client.get("/api/registry", headers={"host": "10.0.0.5:8000"}).status_code == 421
    origins.allow_host("10.0.0.5")
    try:
        assert client.get("/api/registry", headers={"host": "10.0.0.5:8000"}).status_code == 200
    finally:
        origins._allowed.discard("10.0.0.5")


def test_loopback_is_never_removable():
    origins.allow_host("example.com")
    try:
        assert _LOOPBACK_SUBSET <= origins.allowed_hosts()
    finally:
        origins._allowed.discard("example.com")


_LOOPBACK_SUBSET = {"127.0.0.1", "localhost", "::1"}
