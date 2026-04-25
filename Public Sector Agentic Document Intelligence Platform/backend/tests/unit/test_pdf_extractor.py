"""Round-trip the synthetic corpus through extractor + chunker."""
from __future__ import annotations

from app.chunking import chunk_pages, extract_pdf
from app.seed.generate_sample_pdfs import build_sample_pdfs


def test_each_sample_pdf_extracts_real_text() -> None:
    samples = build_sample_pdfs()
    assert len(samples) == 3
    for sample in samples:
        pages = extract_pdf(sample.bytes_)
        assert len(pages) >= 1
        full = "\n\n".join(p.text for p in pages)
        assert len(full) > 500, f"{sample.filename} extracted too little text"
        # A characteristic phrase from each document should be present.
        if "grant" in sample.filename:
            assert "Resilient Communities" in full
        elif "procurement" in sample.filename:
            assert "Procurement Notice" in full
        elif "policy-memo" in sample.filename:
            assert "Public Records" in full


def test_each_sample_pdf_chunks_cleanly() -> None:
    for sample in build_sample_pdfs():
        pages = extract_pdf(sample.bytes_)
        chunks = chunk_pages([(p.page_number, p.text) for p in pages])
        assert chunks, f"{sample.filename} produced zero chunks"
        assert all(c.text.strip() for c in chunks)
        assert all(c.page_start <= c.page_end for c in chunks)
