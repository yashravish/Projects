"""Sentence-aware recursive chunker.

Each chunk records the source page range (`page_start`, `page_end`) and the
absolute character offsets in the concatenated document text (`char_start`,
`char_end`). These offsets are stable so we can highlight back to source.

Algorithm:

1. Concatenate normalized pages with `\n\n` separators, recording each page's
   start offset to compute per-chunk page ranges.
2. Recursively split on a hierarchy of separators (`\n\n`, `\n`, `. `, `; `,
   `, `, ` `) until each fragment is ≤ `max_chars`.
3. Greedily pack fragments into chunks of size `target_chars` with overlap of
   `overlap_chars`, never crossing a fragment boundary.
4. Tokens are estimated cheaply as `len(text) / 4` rounded up — accurate enough
   for cost ledgers; replaced with a real tokenizer at embed time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = 1400
    max_chars: int = 1800
    min_chars: int = 200
    overlap_chars: int = 200
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    index: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str
    token_estimate: int


def _recursive_split(text: str, separators: tuple[str, ...], max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    if not separators:
        # Hard wrap — last resort. Splits inside a word; rare in practice
        # because the separator hierarchy includes ` `.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sep, *rest = separators
    pieces: list[str] = []
    for part in text.split(sep):
        if not part:
            continue
        with_sep = part + sep if part is not text.split(sep)[-1] else part
        if len(with_sep) <= max_chars:
            pieces.append(with_sep)
        else:
            pieces.extend(_recursive_split(with_sep, tuple(rest), max_chars))
    return pieces


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def chunk_pages(pages: list[tuple[int, str]], cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a list of `(page_number, page_text)` tuples.

    Returns chunks in document order. Empty input → empty list.
    """
    cfg = cfg or ChunkConfig()
    if not pages:
        return []

    # Build concatenated document with per-page offsets.
    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []  # (page_number, start_offset)
    cursor = 0
    for page_number, text in pages:
        text = text or ""
        page_starts.append((page_number, cursor))
        parts.append(text)
        cursor += len(text)
        # separator
        parts.append("\n\n")
        cursor += 2
    full_text = "".join(parts).rstrip()
    if not full_text.strip():
        return []

    # Recursive split into atomic fragments.
    fragments = _recursive_split(full_text, cfg.separators, cfg.max_chars)

    # Compute fragment offsets in `full_text` by walking with `find` from cursor.
    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for frag in fragments:
        idx = full_text.find(frag, cursor)
        if idx < 0:
            # Fragment not found verbatim (rare separator-stripping artefacts).
            # Skip to keep offsets monotonic and meaningful.
            continue
        offsets.append((idx, idx + len(frag), frag))
        cursor = idx + len(frag)

    # Pack fragments into chunks.
    chunks: list[Chunk] = []
    current: list[tuple[int, int, str]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        text = "".join(f for _, _, f in current).strip()
        if not text:
            current = []
            current_len = 0
            return
        start = current[0][0]
        end = current[-1][1]
        chunks.append(
            Chunk(
                index=len(chunks),
                page_start=_page_for_offset(start, page_starts),
                page_end=_page_for_offset(max(end - 1, start), page_starts),
                char_start=start,
                char_end=end,
                text=text,
                token_estimate=_estimate_tokens(text),
            )
        )
        # Build overlap tail: trailing characters from the just-flushed chunk.
        # Cap tail size so the next fragment can fit without violating
        # max_chars; a too-large fragment yields no tail.
        max_tail = max(0, cfg.max_chars - cfg.min_chars)
        if cfg.overlap_chars > 0 and max_tail > 0:
            tail = _make_overlap_tail(current, cfg.overlap_chars, max_tail=max_tail)
            current = tail
            current_len = sum(len(f) for _, _, f in tail)
        else:
            current = []
            current_len = 0

    for off in offsets:
        frag = off[2]
        # Force-flush if adding the fragment would exceed max_chars; this
        # preserves the hard ceiling even when fragments are themselves large.
        if current and current_len + len(frag) > cfg.max_chars:
            flush()
        if current_len + len(frag) > cfg.target_chars and current_len >= cfg.min_chars:
            flush()
        current.append(off)
        current_len += len(frag)

    flush()
    # If the very last chunk is below min_chars and there's a previous chunk,
    # merge it back to avoid trailing micro-chunks.
    if len(chunks) >= 2 and len(chunks[-1].text) < cfg.min_chars:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = (prev.text + " " + last.text).strip()
        chunks.append(
            Chunk(
                index=prev.index,
                page_start=prev.page_start,
                page_end=last.page_end,
                char_start=prev.char_start,
                char_end=last.char_end,
                text=merged_text,
                token_estimate=_estimate_tokens(merged_text),
            )
        )
    # Re-index in case we merged.
    return [
        Chunk(
            index=i,
            page_start=c.page_start,
            page_end=c.page_end,
            char_start=c.char_start,
            char_end=c.char_end,
            text=c.text,
            token_estimate=c.token_estimate,
        )
        for i, c in enumerate(chunks)
    ]


def _page_for_offset(offset: int, page_starts: list[tuple[int, int]]) -> int:
    """Return the 1-indexed page number whose start ≤ offset."""
    page_number = page_starts[0][0]
    for pn, start in page_starts:
        if start <= offset:
            page_number = pn
        else:
            break
    return page_number


def _make_overlap_tail(
    fragments: list[tuple[int, int, str]],
    overlap_chars: int,
    *,
    max_tail: int,
) -> list[tuple[int, int, str]]:
    """Take the trailing fragments whose combined length covers `overlap_chars`,
    bounded so the tail itself never exceeds `max_tail` characters. A single
    fragment larger than `max_tail` produces no tail at all — carrying it would
    violate the chunker's hard ceiling on subsequent flushes.
    """
    tail: list[tuple[int, int, str]] = []
    acc = 0
    for off in reversed(fragments):
        flen = len(off[2])
        if not tail and flen > max_tail:
            return []
        if tail and acc + flen > max_tail:
            break
        tail.insert(0, off)
        acc += flen
        if acc >= overlap_chars:
            break
    return tail
