"""Shared pytest setup.

Two concerns, both about isolation the suite otherwise has no way to enforce.
"""
import pytest

from lamplighter import session as session_mod
from lamplighter.backend import origins, state


@pytest.fixture(scope="session", autouse=True)
def _allow_testclient_host() -> None:
    """Starlette's ``TestClient`` addresses the app as ``http://testserver``,
    which the Host allowlist (``backend/origins.py``) correctly refuses — it is
    not loopback. Rather than carve a bypass into shipped code, the *test
    harness* opts in, once, here. The production default stays fail-closed: a
    name has to be allowed explicitly, and only a session binding that interface
    (or this file) does so."""
    origins.allow_host("testserver")


@pytest.fixture(scope="module", autouse=True)
def _no_cross_file_leaks(request):
    """Fail the module that leaves kernel-scoped globals changed.

    Test files run in one process and share `state._current` (the cached
    project) and `session._current` (the live session). A module that sets
    either and doesn't restore it doesn't fail itself — it fails whichever file
    pytest happens to run next, as a mystery about a project it never created.
    That is a genuinely awful hour to spend, and it has already happened once.

    Module-scoped rather than per-test on purpose: a module-scoped fixture may
    legitimately hold state across its own tests, and restoring after each test
    would break that. The contract is only that it cleans up after itself.
    """
    before = (state.get_project(), session_mod._current)
    yield
    after = (state.get_project(), session_mod._current)
    if before != after:
        # Put it back so the report names one culprit rather than cascading.
        state._current, session_mod._current = before
        names = ["state._current", "session._current"]
        changed = [n for n, b, a in zip(names, before, after) if b is not a]
        pytest.fail(
            f"{request.node.name} leaked shared state: {', '.join(changed)}. "
            "Snapshot and restore it in the fixture that sets it — see "
            "tests/test_live_server.py's `live` fixture.",
            pytrace=False,
        )
