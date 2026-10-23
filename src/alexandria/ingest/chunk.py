from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    ord: int
    start_char: int
    end_char: int
    text: str


def chunk_text(text: str, target_tokens: int = 800, overlap_tokens: int = 100) -> list[Chunk]:
    """Chunk ``text`` into windows of whitespace-separated words, with a fixed overlap.

    ``target_tokens`` and ``overlap_tokens`` count words, not model tokens,
    and words are not tokens: English scientific text averages about 1.6
    bge tokens per word. bge-small-en-v1.5 embeds at most 512 tokens (about
    320 such words), so the default 800-word chunk is roughly 1,250 tokens
    and its tail is never embedded; FTS still indexes all of it. See #8.
    """
    if not text:
        return []

    words: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        words.append((start, i, text[start:i]))

    if not words:
        return []

    if overlap_tokens >= target_tokens:
        overlap_tokens = target_tokens // 4

    step = max(1, target_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    ord_ = 0
    idx = 0
    total = len(words)
    while idx < total:
        window = words[idx : idx + target_tokens]
        start_char = window[0][0]
        end_char = window[-1][1]
        chunks.append(
            Chunk(
                ord=ord_,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
            )
        )
        ord_ += 1
        if idx + target_tokens >= total:
            break
        idx += step
    return chunks
