"""The Lamplighter manager decouples session-up from tab-open: constructing
boots (or adopts) the session with no browser side effect; ``.open()`` always
opens (an explicit ask); ``start()``'s implicit open is the only one that
skips when an editor tab is already connected."""
from backend import ws
from lamplighter import session as session_mod
from lamplighter.session import Lamplighter, Session


def _live_session(monkeypatch) -> Lamplighter:
    """A Lamplighter handle that reports running, installed as the module
    singleton — no server thread (in-process, like the other Session tests)."""
    live = Lamplighter.__new__(Lamplighter)
    Session.__init__(live, "127.0.0.1", 8123)
    monkeypatch.setattr(live, "is_running", lambda: True)
    monkeypatch.setattr(session_mod, "_current", live)
    return live


def test_open_always_opens_even_with_a_tab_connected(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("lamplighter.session.webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr(ws.manager, "active", {object()})
    sess = Session("127.0.0.1", 8123)

    assert sess.open() == "http://127.0.0.1:8123"
    assert opened == ["http://127.0.0.1:8123"]  # explicit ask → explicit tab


def test_constructing_adopts_the_running_session(monkeypatch):
    live = _live_session(monkeypatch)

    def boom(self):  # adoption must not boot a second server
        raise AssertionError("tried to boot a second server")

    monkeypatch.setattr(Lamplighter, "_start_server", boom)
    again = Lamplighter(port=9999, persist=False)  # args ignored on adoption
    assert again is live


def test_start_skips_the_implicit_open_when_a_tab_is_connected(monkeypatch):
    _live_session(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr("lamplighter.session.webbrowser.open", lambda url: opened.append(url))

    monkeypatch.setattr(ws.manager, "active", {object()})
    session_mod.start()
    assert opened == []  # tab already connected — no duplicate

    monkeypatch.setattr(ws.manager, "active", set())
    session_mod.start()
    assert opened == ["http://127.0.0.1:8123"]  # tab gone — reopen
