"""Unit tests for `app.governance.pii`.

The detector is regex-driven, so the tests are intentionally compact — they
cover representative samples of every PII kind, ensure precedence is right
when patterns overlap (a credit card number is not also tagged as a phone
number), and check that `redact()` produces text that no longer triggers
detection on a second pass.
"""
from __future__ import annotations

from app.governance import pii


def test_detect_email_simple() -> None:
    findings = pii.detect("Mail me at jane.doe@example.gov for details.")
    kinds = [f.kind for f in findings]
    assert kinds == ["email"]
    assert "@" not in findings[0].preview or findings[0].preview.startswith("j***")


def test_detect_ssn_strict() -> None:
    findings = pii.detect("SSN: 123-45-6789 — keep secret.")
    assert [f.kind for f in findings] == ["ssn"]
    assert findings[0].preview.endswith("6789")


def test_detect_invalid_ssn_areas_skipped() -> None:
    # 000, 666, 9xx area numbers are not valid SSN prefixes.
    assert pii.detect("000-12-3456") == []
    assert pii.detect("666-12-3456") == []
    assert pii.detect("900-12-3456") == []
    # 00 group, 0000 serial also invalid.
    assert pii.detect("123-00-3456") == []
    assert pii.detect("123-45-0000") == []


def test_detect_credit_card_luhn() -> None:
    # Visa test card (passes Luhn).
    text = "Card on file: 4111 1111 1111 1111 expires soon."
    findings = pii.detect(text)
    kinds = [f.kind for f in findings]
    assert "credit_card" in kinds
    assert any(f.preview.endswith("1111") for f in findings)


def test_detect_credit_card_luhn_failure_skipped() -> None:
    text = "Number 4111 1111 1111 1112 — bad checksum."
    findings = pii.detect(text)
    assert [f.kind for f in findings if f.kind == "credit_card"] == []


def test_detect_phone_formats() -> None:
    text = (
        "Reach me at (415) 555-0173 or at 415-555-0174 "
        "or even 415.555.0175."
    )
    kinds = [f.kind for f in pii.detect(text)]
    assert kinds.count("phone") == 3


def test_detect_ipv4() -> None:
    findings = pii.detect("Server is at 10.0.42.255 — internal only.")
    assert [f.kind for f in findings] == ["ipv4"]
    assert findings[0].preview == "***.***.***.255"


def test_detect_bearer_and_jwt() -> None:
    text = (
        "Authorization: Bearer abcdef123456789012345678 "
        "and a token eyJhbGciOiJIUzI1NiJ9.eyJhYmNkZWYxMjM0NTY3ODkw."
        "MTIzNDU2Nzg5MGFiY2RlZg"
    )
    findings = pii.detect(text)
    kinds = [f.kind for f in findings]
    assert "bearer_token" in kinds
    assert "jwt" in kinds


def test_precedence_credit_card_beats_phone() -> None:
    """A 16-digit Luhn-valid number must be tagged credit_card, not phone."""
    findings = pii.detect("4111 1111 1111 1111")
    kinds = [f.kind for f in findings]
    assert kinds == ["credit_card"]


def test_redact_replaces_in_place() -> None:
    text = "Email jane@example.gov, ssn 123-45-6789."
    redacted, findings = pii.redact(text)
    assert "jane@example.gov" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED:email]" in redacted
    assert "[REDACTED:ssn]" in redacted
    assert {f.kind for f in findings} == {"email", "ssn"}
    # Idempotent: redacting again yields no findings.
    _, more = pii.redact(redacted)
    assert more == []


def test_summarize_counts() -> None:
    text = (
        "first@a.com, second@b.com, third@c.com — "
        "and SSN 123-45-6789."
    )
    findings = pii.detect(text)
    summary = pii.summarize(findings)
    assert summary == {"email": 3, "ssn": 1}


def test_empty_input() -> None:
    assert pii.detect("") == []
    redacted, findings = pii.redact("")
    assert redacted == ""
    assert findings == []
