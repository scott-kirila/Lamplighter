"""Shared pytest setup.

Starlette's ``TestClient`` addresses the app as ``http://testserver``, which the
Host allowlist (``backend/origins.py``) correctly refuses — it is not loopback.
Rather than carve a bypass into shipped code, the *test harness* opts in, once,
here. That keeps the production default fail-closed: a name has to be allowed
explicitly, and only a session binding that interface (or this file) does so.
"""
import pytest

from lamplighter.backend import origins


@pytest.fixture(scope="session", autouse=True)
def _allow_testclient_host() -> None:
    origins.allow_host("testserver")
