"""Per-path auth: /mcp bearer-only, /api/* bearer OR cookie, no_auth bypass."""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from alexandria.server import build_http_app
from alexandria.web.auth import COOKIE_NAME, issue_session


def _stub() -> Starlette:
    async def ok(request):
        return JSONResponse({"path": request.url.path})

    return Starlette(
        routes=[
            Route("/mcp", ok, methods=["POST", "GET"]),
            Route("/api/ping", ok, methods=["GET"]),
            Route("/files/{doc_id}", ok, methods=["GET"]),
            Route("/login", ok, methods=["GET", "POST"]),
        ]
    )


@pytest.fixture
def secret() -> bytes:
    return b"z" * 32


@pytest.fixture
def app_with_auth(secret: bytes):
    # Reach past build_http_app to force session_secret into the middleware
    # without requiring a full cfg/conn/lock.
    app = _stub()
    from alexandria.web.middleware import SmartAuthMiddleware

    app.add_middleware(
        SmartAuthMiddleware,
        mcp_token="s3cret",
        session_secret=secret,
        no_auth=False,
    )
    with TestClient(app) as c:
        yield c


# ---- /mcp: bearer-only -----------------------------------------------------


def test_mcp_requires_bearer(app_with_auth):
    r = app_with_auth.post("/mcp", json={})
    assert r.status_code == 401
    assert "missing" in r.json()["error"]


def test_mcp_rejects_cookie_only(app_with_auth, secret):
    cookie = issue_session(secret, max_age_seconds=60)
    app_with_auth.cookies.set(COOKIE_NAME, cookie)
    r = app_with_auth.post("/mcp", json={})
    assert r.status_code == 401


def test_mcp_accepts_bearer(app_with_auth):
    r = app_with_auth.post(
        "/mcp", json={}, headers={"Authorization": "Bearer s3cret"}
    )
    assert r.status_code == 200


# ---- /api/*: bearer OR cookie ---------------------------------------------


def test_api_requires_auth(app_with_auth):
    r = app_with_auth.get("/api/ping")
    assert r.status_code == 401


def test_api_accepts_bearer(app_with_auth):
    r = app_with_auth.get(
        "/api/ping", headers={"Authorization": "Bearer s3cret"}
    )
    assert r.status_code == 200


def test_api_rejects_wrong_bearer(app_with_auth):
    r = app_with_auth.get(
        "/api/ping", headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401


def test_api_accepts_valid_cookie(app_with_auth, secret):
    cookie = issue_session(secret, max_age_seconds=60)
    app_with_auth.cookies.set(COOKIE_NAME, cookie)
    r = app_with_auth.get("/api/ping")
    assert r.status_code == 200


def test_api_rejects_expired_cookie(app_with_auth, secret):
    from alexandria.web.auth import sign_session

    cookie = sign_session(secret, expires_at=1)
    app_with_auth.cookies.set(COOKIE_NAME, cookie)
    r = app_with_auth.get("/api/ping")
    assert r.status_code == 401


# ---- /files/<doc_id>: bearer, cookie, OR signed URL ------------------------


def test_files_requires_auth(app_with_auth):
    r = app_with_auth.get("/files/DOC")
    assert r.status_code == 401


def test_files_accepts_bearer(app_with_auth):
    r = app_with_auth.get(
        "/files/DOC", headers={"Authorization": "Bearer s3cret"}
    )
    assert r.status_code == 200


def test_files_accepts_cookie(app_with_auth, secret):
    cookie = issue_session(secret, max_age_seconds=60)
    app_with_auth.cookies.set(COOKIE_NAME, cookie)
    r = app_with_auth.get("/files/DOC")
    assert r.status_code == 200


def test_files_accepts_valid_signed_url(app_with_auth, secret):
    from alexandria.web.signed_url import build_query

    qs, _ = build_query(secret, "DOC", ttl_seconds=60)
    r = app_with_auth.get(f"/files/DOC?{qs}")
    assert r.status_code == 200


def test_files_rejects_signed_url_for_wrong_doc(app_with_auth, secret):
    from alexandria.web.signed_url import build_query

    qs, _ = build_query(secret, "DOC", ttl_seconds=60)
    r = app_with_auth.get(f"/files/OTHER?{qs}")
    assert r.status_code == 401


def test_files_rejects_expired_signed_url(app_with_auth, secret):
    from alexandria.web.signed_url import sign

    exp, sig = sign(secret, "DOC", ttl_seconds=1, now=1_000_000)
    r = app_with_auth.get(f"/files/DOC?exp={exp}&sig={sig}")
    assert r.status_code == 401


def test_files_rejects_tampered_signature(app_with_auth, secret):
    from alexandria.web.signed_url import build_query

    qs, _ = build_query(secret, "DOC", ttl_seconds=60)
    # Flip a couple of chars in the signature.
    bad = qs.replace("sig=", "sig=AAAA")
    r = app_with_auth.get(f"/files/DOC?{bad}")
    assert r.status_code == 401


# ---- open paths + no_auth bypass ------------------------------------------


def test_login_is_anonymous(app_with_auth):
    r = app_with_auth.get("/login")
    assert r.status_code == 200


def test_no_auth_lets_everything_through():
    app = build_http_app(auth_token=None, no_auth=True, inner=_stub())
    with TestClient(app) as c:
        assert c.post("/mcp", json={}).status_code == 200
        assert c.get("/api/ping").status_code == 200
        assert c.get("/files/DOC").status_code == 200
