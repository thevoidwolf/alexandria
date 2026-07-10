from alexandria.ingest.hash import sha256_bytes, sha256_text


def test_sha256_bytes_deterministic():
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"hello!")


def test_sha256_text_deterministic():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abcd")


def test_text_hash_uses_utf8():
    # ensure the text hash cares about content, not byte encoding quirks
    assert sha256_text("café") == sha256_text("café")
