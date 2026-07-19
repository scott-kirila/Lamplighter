"""One way to do each thing: Lamplighter() boots (or adopts) the session with
no browser side effect; .open() is the single way a tab opens — a local browser
normally, a reach-the-app recipe when the kernel is remote. `remote` is an
explicit constructor truth (None = auto-detect)."""
from lamplighter import session as session_mod
from lamplighter.session import Lamplighter, Session


def _trap_browser(monkeypatch) -> list[str]:
    opened: list[str] = []
    monkeypatch.setattr("lamplighter.session.webbrowser.open", lambda url: opened.append(url))
    return opened


def test_open_opens_a_tab(monkeypatch):
    opened = _trap_browser(monkeypatch)
    monkeypatch.setattr(session_mod, "_likely_remote", lambda: False)

    assert Session("127.0.0.1", 8123).open() == "http://127.0.0.1:8123"
    assert opened == ["http://127.0.0.1:8123"]


def test_open_on_a_remote_kernel_prints_how_to_reach_the_app(monkeypatch, capsys):
    opened = _trap_browser(monkeypatch)
    monkeypatch.setattr(session_mod, "_likely_remote", lambda: True)

    assert Session("127.0.0.1", 8123).open() == "http://127.0.0.1:8123"  # URL still returned
    assert opened == []  # a browser on the server wouldn't reach the user
    out = capsys.readouterr().out
    assert "ssh -L 8123:127.0.0.1:8123" in out
    assert "http://127.0.0.1:8123" in out
    assert "remote=False" in out  # the misfire escape hatch documents itself


def test_explicit_remote_beats_detection_both_ways(monkeypatch, capsys):
    opened = _trap_browser(monkeypatch)

    monkeypatch.setattr(session_mod, "_likely_remote", lambda: True)
    Session("127.0.0.1", 8123, remote=False).open()  # user says local — open
    assert opened == ["http://127.0.0.1:8123"]

    monkeypatch.setattr(session_mod, "_likely_remote", lambda: False)
    Session("127.0.0.1", 8123, remote=True).open()  # user says remote — print
    assert opened == ["http://127.0.0.1:8123"]  # unchanged
    assert "ssh -L" in capsys.readouterr().out


def test_remote_detection_signals(monkeypatch):
    for var in ("SSH_CONNECTION", "SSH_TTY", "BROWSER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 52422 10.0.0.2 22")
    assert session_mod._likely_remote() is True
    # An explicit BROWSER means the user configured how opening works — trust it.
    monkeypatch.setenv("BROWSER", "firefox")
    assert session_mod._likely_remote() is False


def test_non_local_host_warns_about_the_unauthenticated_server():
    import warnings

    import pytest

    # The server carries no auth — binding beyond localhost must say so loudly.
    with pytest.warns(UserWarning, match="no authentication"):
        session_mod._warn_if_exposed("0.0.0.0")
    # The local hosts stay silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for host in ("127.0.0.1", "localhost", "::1"):
            session_mod._warn_if_exposed(host)


def test_constructing_adopts_the_running_session(monkeypatch):
    live = Lamplighter.__new__(Lamplighter)
    Session.__init__(live, "127.0.0.1", 8123)
    monkeypatch.setattr(live, "is_running", lambda: True)
    monkeypatch.setattr(session_mod, "_current", live)

    def boom(self):  # adoption must not boot a second server
        raise AssertionError("tried to boot a second server")

    monkeypatch.setattr(Lamplighter, "_start_server", boom)
    again = Lamplighter(port=9999, persist=False)  # args ignored on adoption
    assert again is live
