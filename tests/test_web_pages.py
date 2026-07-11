"""Server-rendered pages: /login flow, / landing, /logout."""
from __future__ import annotations

import threading

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from alexandria.web.api import build_api_routes
from alexandria.web.auth import (
    COOKIE_NAME,
    hash_password,
    issue_session,
    write_password_hash,
)
from alexandria.web.jobs import JobQueue
from alexandria.web.middleware import SmartAuthMiddleware
from alexandria.web.pages import build_page_routes


@pytest.fixture
def secret() -> bytes:
    return b"z" * 32


@pytest.fixture
def jobs(cfg, conn):
    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


@pytest.fixture
def app(cfg, conn, jobs, secret, fake_embed):
    lock = threading.Lock()
    routes = build_api_routes(cfg, conn, lock, jobs=jobs) \
        + build_page_routes(cfg, secret, conn, lock, jobs=jobs)
    a = Starlette(routes=routes)
    a.add_middleware(
        SmartAuthMiddleware,
        mcp_token="mcp-t0k",
        session_secret=secret,
        no_auth=False,
    )
    return a


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def logged_in(client, secret, cfg):
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    client.post("/login", data={"password": "pw"})
    return client


# ---- login GET / POST ------------------------------------------------------


def test_login_get_renders_form(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "password" in r.text.lower()
    assert 'name="password"' in r.text


def test_login_post_no_password_set_returns_503(client, cfg):
    # Ensure no hash file present.
    p = cfg.web_password_hash_path
    if p.exists():
        p.unlink()
    r = client.post("/login", data={"password": "anything"}, follow_redirects=False)
    assert r.status_code == 503
    assert "set-web-password" in r.text


def test_login_post_wrong_password_returns_401(client, cfg):
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    r = client.post("/login", data={"password": "wrong"}, follow_redirects=False)
    assert r.status_code == 401
    assert "Invalid password" in r.text


def test_login_post_correct_password_sets_cookie_and_redirects(client, cfg):
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    r = client.post("/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    set_cookie = r.headers["set-cookie"]
    assert set_cookie.startswith(f"{COOKIE_NAME}=")
    assert "httponly" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()


# ---- logout ---------------------------------------------------------------


def test_logout_clears_cookie(client, secret):
    client.cookies.set(COOKIE_NAME, issue_session(secret, 3600))
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
    assert 'max-age=0' in r.headers["set-cookie"].lower() or \
           'expires=' in r.headers["set-cookie"].lower()


# ---- landing page --------------------------------------------------------


def test_landing_unauth_redirects_to_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_landing_with_session_renders(logged_in):
    r = logged_in.get("/")
    assert r.status_code == 200
    assert "Corpus" in r.text
    assert "Drop PDFs" in r.text
    assert "Recent activity" in r.text


def test_landing_with_bearer_also_renders(client):
    r = client.get(
        "/", headers={"Authorization": "Bearer mcp-t0k"}, follow_redirects=False,
    )
    assert r.status_code == 200


def test_landing_shows_recent_jobs(logged_in, jobs):
    jobs.enqueue_url("https://example.com/foo", None, [])
    r = logged_in.get("/")
    assert r.status_code == 200
    assert "https://example.com/foo" in r.text


# ---- middleware redirect for arbitrary page paths ------------------------


def test_unknown_page_route_redirects_when_unauth(client):
    r = client.get("/documents/nope", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
