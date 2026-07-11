"""Password hashing (bcrypt) and stateless signed-cookie sessions.

Sessions are a cookie value of the form ``<expires_at>.<sig>`` where the
signature is an HMAC-BLAKE2b of the expiry timestamp under a per-install
secret. No server-side store — revocation is by rotating the secret.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

import bcrypt

from alexandria.config import Config

COOKIE_NAME = "alx_session"


# ---- password hashing -------------------------------------------------------


def hash_password(password: str, *, rounds: int = 12) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds))


def verify_password(password: str, stored_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    except ValueError:
        return False


def write_password_hash(path: Path, hashed: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hashed)
    os.chmod(path, 0o600)


def read_password_hash(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes().strip() or None


# ---- session secret ---------------------------------------------------------


def load_or_create_session_secret(path: Path) -> bytes:
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 16:
            return data
    secret = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(secret)
    os.chmod(path, 0o600)
    return secret


def get_or_create_session_secret(cfg: Config) -> bytes:
    return load_or_create_session_secret(cfg.web_session_secret_path)


# ---- session cookie sign/verify --------------------------------------------


def _b64u_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def sign_session(secret: bytes, expires_at: int) -> str:
    payload = str(expires_at).encode("ascii")
    sig = hmac.new(secret, payload, hashlib.blake2b).digest()
    return f"{expires_at}.{_b64u_encode(sig)}"


def issue_session(secret: bytes, max_age_seconds: int, now: int | None = None) -> str:
    if now is None:
        now = int(time.time())
    return sign_session(secret, now + max_age_seconds)


def verify_session(secret: bytes, cookie_value: str, now: int | None = None) -> bool:
    if now is None:
        now = int(time.time())
    if not cookie_value or "." not in cookie_value:
        return False
    try:
        exp_str, sig_b64 = cookie_value.split(".", 1)
        exp = int(exp_str)
    except ValueError:
        return False
    if now > exp:
        return False
    expected = hmac.new(secret, exp_str.encode("ascii"), hashlib.blake2b).digest()
    try:
        got = _b64u_decode(sig_b64)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected, got)
