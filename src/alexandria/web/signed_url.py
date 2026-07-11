"""HMAC-signed URLs for time-limited access to /files/<doc_id>.

Signature is HMAC-SHA256 over ``f"{doc_id}|{exp}"`` under the same
per-install ``session_secret`` used by browser session cookies. That secret
already lives at ``$ALEXANDRIA_HOME/session_secret`` (chmod 600) and gets
rotated the same way cookies do — no extra key material to manage.

The query string carries ``exp`` (unix seconds) and ``sig`` (urlsafe-b64
without padding). Callers stitch them onto ``/files/<doc_id>``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time

_DEFAULT_TTL = 3600
_MAX_TTL = 86400


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _sig(secret: bytes, doc_id: str, exp: int) -> str:
    payload = f"{doc_id}|{exp}".encode("ascii")
    return _b64u(hmac.new(secret, payload, hashlib.sha256).digest())


def clamp_ttl(ttl_seconds: int | None) -> int:
    """Clamp the requested TTL to [1, MAX_TTL], defaulting to 1h."""
    ttl = _DEFAULT_TTL if ttl_seconds is None else int(ttl_seconds)
    if ttl < 1:
        return 1
    if ttl > _MAX_TTL:
        return _MAX_TTL
    return ttl


def sign(secret: bytes, doc_id: str, ttl_seconds: int | None = None,
         now: int | None = None) -> tuple[int, str]:
    """Return (exp, sig) for a URL that grants access to ``doc_id``."""
    ttl = clamp_ttl(ttl_seconds)
    exp = (int(time.time()) if now is None else now) + ttl
    return exp, _sig(secret, doc_id, exp)


def verify(secret: bytes, doc_id: str, exp: int | str, sig: str,
           now: int | None = None) -> bool:
    """Constant-time verify. Rejects expired or malformed signatures."""
    try:
        exp_int = int(exp)
    except (TypeError, ValueError):
        return False
    if (int(time.time()) if now is None else now) > exp_int:
        return False
    expected = _sig(secret, doc_id, exp_int)
    return hmac.compare_digest(expected.encode("ascii"), sig.encode("ascii"))


def build_query(secret: bytes, doc_id: str, ttl_seconds: int | None = None,
                now: int | None = None) -> tuple[str, int]:
    """Return ("exp=...&sig=...", exp) so callers can stitch it onto a URL."""
    exp, sig = sign(secret, doc_id, ttl_seconds, now)
    return f"exp={exp}&sig={sig}", exp
