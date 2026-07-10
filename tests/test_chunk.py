from alexandria.ingest.chunk import chunk_text


def test_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_single_short_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, target_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_windowed_with_overlap():
    words = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(words, target_tokens=8, overlap_tokens=2)
    # step = 8 - 2 = 6 → windows starting at 0, 6, 12 (then remainder <8 fits in last)
    assert len(chunks) >= 3
    # ord is 0-based and monotonic
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    # char offsets bracket the source text
    for c in chunks:
        assert words[c.start_char:c.end_char] == c.text


def test_overlap_clamped_when_larger_than_target():
    chunks = chunk_text("a b c d e f g h", target_tokens=4, overlap_tokens=10)
    # Should still produce non-empty output without infinite loop.
    assert len(chunks) > 0
