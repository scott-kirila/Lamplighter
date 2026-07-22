"""Caching policy for the served UI.

The upgrade path depends on this. Vite content-hashes everything under
``assets/`` and deletes the previous build's files, so a cached ``index.html``
that survives an upgrade points at a script that 404s — the tab renders blank
and only a hard refresh fixes it. The two halves therefore need opposite
policies, and nothing else in the suite would notice if they regressed.
"""
import pytest
from fastapi.testclient import TestClient

from lamplighter.backend.app import app
from lamplighter.backend.dist import frontend_dist

pytestmark = pytest.mark.skipif(
    frontend_dist() is None,
    reason="no built frontend — the static mount isn't registered (CI backend job)",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_index_is_revalidated_on_every_load(client):
    resp = client.get("/")
    assert resp.status_code == 200
    # "no-cache" means revalidate, not "don't store" — the ETag below keeps it cheap.
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers.get("etag")


def test_index_revalidation_is_a_cheap_304(client):
    etag = client.get("/").headers["etag"]
    assert client.get("/", headers={"If-None-Match": etag}).status_code == 304


def test_hashed_assets_are_cached_immutably(client):
    dist = frontend_dist()
    asset = next((p for p in (dist / "assets").iterdir() if p.suffix == ".js"), None)
    assert asset is not None, "expected at least one built JS asset"
    resp = client.get(f"/assets/{asset.name}")
    assert resp.status_code == 200
    cc = resp.headers["cache-control"]
    assert "immutable" in cc and "max-age=31536000" in cc
