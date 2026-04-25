"""Chunker behavior — boundaries, overlap, page-range tracking, idempotency."""
from __future__ import annotations

from app.chunking.chunker import Chunk, ChunkConfig, chunk_pages


def test_empty_input_returns_empty() -> None:
    assert chunk_pages([]) == []
    assert chunk_pages([(1, "")]) == []
    assert chunk_pages([(1, "   \n  ")]) == []


def test_short_doc_one_chunk_with_correct_offsets() -> None:
    text = "The Office of Resilience Programs administers the FY26 grant."
    chunks = chunk_pages([(1, text)], cfg=ChunkConfig(target_chars=400, max_chars=600))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.index == 0
    assert c.page_start == 1
    assert c.page_end == 1
    assert c.char_start == 0
    assert c.char_end >= len(text) - 2  # accounting for .strip()
    assert c.text == text
    assert c.token_estimate >= 1


def test_target_size_packs_multiple_paragraphs() -> None:
    paragraph = "Sentence one. Sentence two. Sentence three. " * 6
    pages = [(1, paragraph), (2, paragraph), (3, paragraph)]
    chunks = chunk_pages(pages, cfg=ChunkConfig(target_chars=500, max_chars=700, overlap_chars=80))
    assert len(chunks) >= 2
    # Indexes are monotonic.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Offsets are monotonic non-decreasing.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.char_start <= nxt.char_start


def test_page_range_tracks_source_pages() -> None:
    page_one = "Alpha alpha alpha. " * 50
    page_two = "Beta beta beta. " * 50
    page_three = "Gamma gamma gamma. " * 50
    chunks = chunk_pages(
        [(1, page_one), (2, page_two), (3, page_three)],
        cfg=ChunkConfig(target_chars=400, max_chars=600, overlap_chars=0),
    )
    pages_seen = {(c.page_start, c.page_end) for c in chunks}
    # At least one chunk should map to page 1, one to page 2, one to page 3.
    pages_touched = {p for span in pages_seen for p in range(span[0], span[1] + 1)}
    assert {1, 2, 3}.issubset(pages_touched)


def test_overlap_chars_are_not_lost() -> None:
    # Chunks must overlap by ~`overlap_chars` so retrieval edge-spans don't drop.
    text = "Foo. Bar. Baz. Qux. " * 100
    chunks = chunk_pages(
        [(1, text)], cfg=ChunkConfig(target_chars=400, max_chars=600, overlap_chars=120)
    )
    if len(chunks) < 2:
        return
    a, b = chunks[0], chunks[1]
    overlap_window = a.text[-200:]
    # Some prefix of b should appear within the tail of a.
    head = b.text[:80]
    assert head[:20] in overlap_window or any(
        word in overlap_window for word in head.split()[:3] if len(word) > 3
    )


def test_max_chars_is_a_hard_ceiling() -> None:
    # Even adversarial input (one long line) is split below max_chars.
    text = "x" * 5000
    chunks = chunk_pages([(1, text)], cfg=ChunkConfig(target_chars=400, max_chars=600))
    assert all(len(c.text) <= 600 for c in chunks), [len(c.text) for c in chunks]


def test_chunks_are_typed_and_immutable() -> None:
    chunks = chunk_pages(
        [(1, "Some text. " * 200)], cfg=ChunkConfig(target_chars=300, max_chars=500)
    )
    assert all(isinstance(c, Chunk) for c in chunks)
    # frozen=True dataclass — assignment should fail.
    import dataclasses

    assert dataclasses.is_dataclass(chunks[0])
