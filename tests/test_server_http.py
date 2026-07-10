"""Tests for the M7 Streamable HTTP transport layer.

We exercise the auth middleware and the bind safety rail with a stub inner
Starlette app rather than the real FastMCP one — the FastMCP session manager
can only start once per process, so reusing it across tests would fail on the
second boot. Auth is transport-level and doesn't care what's downstream.
"""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from alexandria.server import build_http_app, validate_bind


def _stub_inner() -> Starlette:
    async def ok(_request):
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/mcp", ok, methods=["POST", "GET"])])


# ---- safety rail ------------------------------------------------------------


def test_safety_rail_refuses_non_loopback_without_auth():
    with pytest.raises(ValueError, match="refusing to bind non-loopback"):
        validate_bind("0.0.0.0", auth_token=None, no_auth=False)


def test_safety_rail_allows_non_loopback_with_auth():
    validate_bind("0.0.0.0", auth_token="secret", no_auth=False)


def test_safety_rail_allows_non_loopback_with_explicit_no_auth():
    validate_bind("0.0.0.0", auth_token=None, no_auth=True)


def test_safety_rail_allows_loopback_bare():
    for h in ("127.0.0.1", "localhost", "::1"):
        validate_bind(h, auth_token=None, no_auth=False)


# ---- auth middleware --------------------------------------------------------


@pytest.fixture
def auth_client():
    app = build_http_app(auth_token="s3cret", no_auth=False, inner=_stub_inner())
    with TestClient(app) as client:
        yield client


def test_missing_bearer_header_returns_401(auth_client):
    r = auth_client.post("/mcp", json={})
    assert r.status_code == 401
    assert "missing" in r.json()["error"]


def test_wrong_bearer_returns_401(auth_client):
    r = auth_client.post(
        "/mcp", json={}, headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401
    assert "invalid" in r.json()["error"]


def test_non_bearer_scheme_returns_401(auth_client):
    r = auth_client.post(
        "/mcp", json={}, headers={"Authorization": "Basic s3cret"}
    )
    assert r.status_code == 401


def test_correct_bearer_reaches_inner_app(auth_client):
    r = auth_client.post(
        "/mcp", json={}, headers={"Authorization": "Bearer s3cret"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_no_auth_mode_lets_anonymous_through():
    app = build_http_app(auth_token=None, no_auth=True, inner=_stub_inner())
    with TestClient(app) as client:
        r = client.post("/mcp", json={})
        assert r.status_code == 200
