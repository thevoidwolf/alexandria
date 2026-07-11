"""HMAC signed-URL helpers for /files/<doc_id>."""
from __future__ import annotations

import pytest

from alexandria.web.signed_url import build_query, clamp_ttl, sign, verify


SECRET = b"z" * 32


def test_sign_and_verify_roundtrip():
    exp, sig = sign(SECRET, "DOC123", ttl_seconds=60)
    assert verify(SECRET, "DOC123", exp, sig)


def test_verify_rejects_tampered_doc_id():
    exp, sig = sign(SECRET, "DOC123", ttl_seconds=60)
    assert not verify(SECRET, "OTHER", exp, sig)


def test_verify_rejects_wrong_secret():
    exp, sig = sign(SECRET, "DOC123", ttl_seconds=60)
    assert not verify(b"different-secret" * 2, "DOC123", exp, sig)


def test_verify_rejects_expired():
    exp, sig = sign(SECRET, "DOC123", ttl_seconds=1, now=1_000_000)
    assert not verify(SECRET, "DOC123", exp, sig, now=1_000_100)


def test_verify_rejects_malformed():
    assert not verify(SECRET, "DOC", "notanumber", "sig")
    assert not verify(SECRET, "DOC", None, "sig")
    assert not verify(SECRET, "DOC", 123, "")


@pytest.mark.parametrize(
    "given,expected",
    [
        (None, 3600),
        (0, 1),
        (-5, 1),
        (60, 60),
        (86400, 86400),
        (99999, 86400),
    ],
)
def test_clamp_ttl(given, expected):
    assert clamp_ttl(given) == expected


def test_build_query_shape():
    qs, exp = build_query(SECRET, "DOC", ttl_seconds=60)
    assert qs.startswith("exp=") and "&sig=" in qs
    assert str(exp) in qs
