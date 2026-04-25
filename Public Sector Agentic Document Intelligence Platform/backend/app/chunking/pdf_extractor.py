"""PDF text extraction.

Primary path: PyMuPDF (`fitz`) which is fast and handles most government PDFs well.
Fallback path: pdfplumber for cases where PyMuPDF returns suspiciously little
text (heuristic: <20 chars / page average), which is a signal of column-based or
table-heavy layouts.

Returns one `ExtractedPage` per page with a normalized text and the source
1-indexed page number.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import cast

import fitz
import pdfplumber

from app.logging_config import get_logger

log = get_logger("pdf_extractor")


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int  # 1-indexed
    text: str


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"-\n(?=[a-z])")


def _normalize(raw: str) -> str:
    """Normalize whitespace and stitch hyphenated line-breaks."""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_pymupdf(data: bytes) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            raw = cast(str, page.get_text("text"))
            pages.append(ExtractedPage(page_number=i, text=_normalize(raw)))
    return pages


def _extract_pdfplumber(data: bytes) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append(ExtractedPage(page_number=i, text=_normalize(raw)))
    return pages


def extract_pdf(data: bytes) -> list[ExtractedPage]:
    """Extract one `ExtractedPage` per page from a PDF byte stream.

    PyMuPDF first; if average non-empty text per page is suspiciously low we
    re-extract via pdfplumber and keep whichever total is larger.
    """
    primary = _extract_pymupdf(data)
    total_chars = sum(len(p.text) for p in primary)
    avg = total_chars / max(len(primary), 1)
    if avg >= 20 or not primary:
        log.info(
            "pdf.extract",
            backend="pymupdf",
            pages=len(primary),
            chars=total_chars,
        )
        return primary

    log.info(
        "pdf.extract.fallback",
        reason="low_avg_chars",
        avg=round(avg, 2),
        pages=len(primary),
    )
    fallback = _extract_pdfplumber(data)
    fallback_chars = sum(len(p.text) for p in fallback)
    if fallback_chars > total_chars:
        log.info(
            "pdf.extract",
            backend="pdfplumber",
            pages=len(fallback),
            chars=fallback_chars,
        )
        return fallback
    return primary
