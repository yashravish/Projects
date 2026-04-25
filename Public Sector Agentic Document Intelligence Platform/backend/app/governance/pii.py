"""Regex-based PII detection and redaction.

The detector is deterministic and operates on raw text. It returns a list
of `Finding` records describing the location, kind, and a short masked
preview of every match. `redact()` returns the redacted text plus the
findings, so callers (the inquiry pipeline, the audit emitter) can
record *what* was redacted without persisting *the actual values*.

Categories detected (in this order of precedence):

  * `email`           — RFC-5322-lite local@domain
  * `ssn`             — `NNN-NN-NNNN`
  * `credit_card`     — 13–19 digits with optional separators, Luhn-checked
  * `phone`           — North-American 10-digit numbers in common formats
  * `ipv4`            — dotted quad with octets ≤ 255
  * `bearer_token`    — Authorization-style `Bearer <token>` strings
  * `jwt`             — three base64url segments separated by dots

The order matters because, e.g., a credit card detected first must not
also be flagged as a "phone number" by the looser phone matcher. Each
match is stripped from the active scan window before subsequent matchers
run.

Limitations (documented for the future Stage 7 PII vendor integration):

  * No NER — we don't detect proper names, addresses, dates of birth.
  * No language awareness — every text is treated as English-encoded.
  * Luhn validation is the only "smart" check; the rest are pure regex.

The detector is fast: a single pass per category, all categories together
process a 100 kB document in well under 50 ms on CPython 3.12.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable

PII_KIND_PRECEDENCE: tuple[str, ...] = (
    "email",
    "ssn",
    "credit_card",
    "phone",
    "ipv4",
    "bearer_token",
    "jwt",
)


@dataclasses.dataclass(frozen=True)
class Finding:
    """One PII detection.

    `start` and `end` are character offsets in the *original* text.
    `preview` is a masked rendering safe to record in the audit log.
    """

    kind: str
    start: int
    end: int
    preview: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "preview": self.preview,
        }


# ── Regex sources ────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# 13-19 digit groups with optional - or space separators every 4 digits.
_CARD_RE = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b"
)

# (123) 456-7890, 123-456-7890, 123.456.7890, 123 456 7890, 1234567890
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
_BEARER_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9._\-+/=]{16,}", re.IGNORECASE
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)

_KIND_TO_RE: dict[str, re.Pattern[str]] = {
    "email": _EMAIL_RE,
    "ssn": _SSN_RE,
    "credit_card": _CARD_RE,
    "phone": _PHONE_RE,
    "ipv4": _IPV4_RE,
    "bearer_token": _BEARER_RE,
    "jwt": _JWT_RE,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn check on a digits-only string."""
    total = 0
    n = len(digits)
    if n < 13 or n > 19:
        return False
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - ord("0")
        if d < 0 or d > 9:
            return False
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_preview(kind: str, raw: str) -> str:
    """Render a short, safe preview for audit records.

    Examples:
        email     → `j***e@e***e.com`
        ssn       → `***-**-1234`
        credit_card → `**** **** **** 1234`
        phone     → `(***) ***-7890`
        ipv4      → `***.***.***.42`
        bearer_token → `Bearer ****`
        jwt       → `eyJ****.****.****`
    """
    if kind == "email":
        local, _, domain = raw.partition("@")
        local_m = (local[0] + "***" + local[-1]) if len(local) >= 2 else "***"
        if "." in domain:
            base, _, tld = domain.rpartition(".")
            base_m = (base[0] + "***" + base[-1]) if len(base) >= 2 else "***"
            return f"{local_m}@{base_m}.{tld}"
        return f"{local_m}@***"
    if kind == "ssn":
        return f"***-**-{raw[-4:]}"
    if kind == "credit_card":
        digits = re.sub(r"\D", "", raw)
        return f"**** **** **** {digits[-4:]}"
    if kind == "phone":
        digits = re.sub(r"\D", "", raw)
        return f"(***) ***-{digits[-4:]}"
    if kind == "ipv4":
        last = raw.rsplit(".", 1)[-1]
        return f"***.***.***.{last}"
    if kind == "bearer_token":
        return "Bearer ****"
    if kind == "jwt":
        return "eyJ****.****.****"
    return "****"


# ── Public API ───────────────────────────────────────────────────────────────


def detect(text: str) -> list[Finding]:
    """Return all PII findings in `text`, sorted by start offset.

    Findings never overlap: a higher-precedence kind that matched a span
    suppresses lower-precedence matchers from claiming the same characters.
    """
    if not text:
        return []
    claimed: list[tuple[int, int]] = []
    findings: list[Finding] = []

    def _is_claimed(start: int, end: int) -> bool:
        for s, e in claimed:
            if start < e and end > s:
                return True
        return False

    for kind in PII_KIND_PRECEDENCE:
        pattern = _KIND_TO_RE[kind]
        for m in pattern.finditer(text):
            s, e = m.span()
            if _is_claimed(s, e):
                continue
            value = m.group(0)
            if kind == "credit_card":
                digits = re.sub(r"\D", "", value)
                if not _luhn_valid(digits):
                    continue
            findings.append(
                Finding(kind=kind, start=s, end=e, preview=_mask_preview(kind, value))
            )
            claimed.append((s, e))

    findings.sort(key=lambda f: (f.start, f.end))
    return findings


def redact(text: str, *, mask: str = "[REDACTED:{kind}]") -> tuple[str, list[Finding]]:
    """Return `(redacted_text, findings)`.

    Replacements happen right-to-left so character offsets in `findings`
    remain valid against the *original* text. The default mask preserves
    the kind for downstream consumers (audit records, frontend chips).
    """
    findings = detect(text)
    if not findings:
        return text, []
    out = text
    for f in reversed(findings):
        replacement = mask.format(kind=f.kind)
        out = out[: f.start] + replacement + out[f.end :]
    return out, findings


def summarize(findings: Iterable[Finding]) -> dict[str, int]:
    """Tallied counts by kind — convenient for audit metadata."""
    out: dict[str, int] = {}
    for f in findings:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out


__all__ = [
    "Finding",
    "PII_KIND_PRECEDENCE",
    "detect",
    "redact",
    "summarize",
]
