"""PII detection (`pii.py`) and tenant-scoped retention purges (`retention.py`).

Callers must enforce `organization_id`. `pii` is regex-based (SSN, email,
phone, card, IP, JWT-shaped tokens) for the redaction/audit path; it is not a
full DLP substitute. `retention` returns row counts; zero TTL means retain
forever.
"""
from __future__ import annotations
