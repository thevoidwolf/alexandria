"""Password hashing + signed-session-cookie roundtrips."""
from __future__ import annotations

import os
import time
from pathlib import Path

from alexandria.web.auth import (
    hash_password,
    issue_session,
    load_or_create_session_secret,
    read_password_hash,
    sign_session,
    verify_password,
    verify_session,
    write_password_hash,
)


# ---- password --------------------------------------------------------------


def test_password_roundtrip():
    hashed = hash_password("hunter2", rounds=4)
    assert verify_password("hunter2", hashed)
    assert not verify_password("hunter3", hashed)


def test_verify_password_tolerates_garbage_hash():
    assert not verify_password("anything", b"not-a-bcrypt-hash")


def test_write_password_hash_chmod_600(tmp_path: Path):
    p = tmp_path / "sub" / "hash"
    write_password_hash(p, hash_password("x", rounds=4))
    mode = os.stat(p).st_mode & 0o777
    assert mode == 0o600
    stored = read_password_hash(p)
    assert stored is not None
    assert verify_password("x", stored)


def test_read_password_hash_missing_returns_none(tmp_path: Path):
    assert read_password_hash(tmp_path / "nope") is None


# ---- session cookie --------------------------------------------------------


def test_session_roundtrip():
    secret = b"a" * 32
    now = 1_000_000
    cookie = issue_session(secret, max_age_seconds=60, now=now)
    assert verify_session(secret, cookie, now=now + 30)


def test_session_rejects_expired():
    secret = b"a" * 32
    cookie = sign_session(secret, expires_at=100)
    assert not verify_session(secret, cookie, now=200)


def test_session_rejects_tampered_expiry():
    secret = b"a" * 32
    cookie = sign_session(secret, expires_at=100)
    _, sig = cookie.split(".", 1)
    tampered = f"999999999.{sig}"
    assert not verify_session(secret, tampered, now=50)


def test_session_rejects_wrong_secret():
    cookie = sign_session(b"a" * 32, expires_at=int(time.time()) + 60)
    assert not verify_session(b"b" * 32, cookie)


def test_session_rejects_malformed_cookie():
    secret = b"a" * 32
    for bad in ("", "no-dot", "abc.def", "1.$$$"):
        assert not verify_session(secret, bad, now=50)


def test_session_secret_persisted(tmp_path: Path):
    p = tmp_path / "secret"
    s1 = load_or_create_session_secret(p)
    s2 = load_or_create_session_secret(p)
    assert s1 == s2
    assert len(s1) >= 16
    assert os.stat(p).st_mode & 0o777 == 0o600
