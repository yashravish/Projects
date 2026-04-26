from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from app.core.config import get_settings

try:  # OpenAI is optional
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class AnalysisResult:
    root_cause_summary: str
    likely_file_or_component: str
    suggested_fix: str
    severity: str
    confidence_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class MockAnalyzer:
    PATTERNS: list[tuple[re.Pattern[str], tuple[str, str, str, str, float]]] = [
        (
            re.compile(r"ModuleNotFoundError:?\s*No module named '(\S+)'", re.I),
            (
                "Missing Python package causing import failure.",
                "requirements.txt / {0}",
                "Add the module to your dependencies (pip/uv/poetry) and reinstall the environment.",
                "medium",
                0.78,
            ),
        ),
        (
            re.compile(r"ECONNREFUSED|connection refused", re.I),
            (
                "Network connection refused; downstream dependency unavailable.",
                "client / service endpoint",
                "Verify the service is up, check host/port, firewall, and health checks.",
                "high",
                0.72,
            ),
        ),
        (
            re.compile(r"timeout|ETIMEDOUT|DeadlineExceeded", re.I),
            (
                "Operation timed out; likely slow dependency or low timeout budget.",
                "HTTP client or database pool",
                "Increase timeouts carefully, add retries with backoff, and profile slow calls.",
                "medium",
                0.7,
            ),
        ),
        (
            re.compile(r"SyntaxError|IndentationError", re.I),
            (
                "Parse error in source; invalid Python syntax.",
                "reported file in traceback",
                "Open the file at the line indicated and fix the syntax/indentation.",
                "low",
                0.9,
            ),
        ),
    ]

    def analyze(self, logs: str) -> AnalysisResult:
        for rx, (summary, comp, fix, sev, conf) in self.PATTERNS:
            m = rx.search(logs)
            if m:
                comp_fmt = comp.format(m.group(1)) if m.groups() else comp
                return AnalysisResult(
                    root_cause_summary=summary,
                    likely_file_or_component=comp_fmt,
                    suggested_fix=fix,
                    severity=sev,
                    confidence_score=conf,
                )
        # Default heuristic
        h = int(hashlib.sha1(logs.encode("utf-8", errors="replace")).hexdigest()[:4], 16) % 100
        conf = 0.4 + (h / 200)
        sev = "low" if h < 40 else "medium" if h < 75 else "high"
        return AnalysisResult(
            root_cause_summary="Generic build/test failure; see logs for failing command output.",
            likely_file_or_component="ci / pipeline",
            suggested_fix="Re-run the failing stage locally, capture the first error line, and fix the underlying test or build step.",
            severity=sev,
            confidence_score=round(min(0.95, conf), 2),
        )


def _openai_analyze(logs: str) -> AnalysisResult:
    s = get_settings()
    if not s.openai_api_key or OpenAI is None:
        raise ValueError("OpenAI not configured")
    client = OpenAI(api_key=s.openai_api_key)
    prompt = (
        "You are a senior SRE. Given CI/CD logs, respond ONLY with a JSON object: "
        '{"root_cause_summary":string,"likely_file_or_component":string,'
        '"suggested_fix":string,"severity":"low|medium|high|critical","confidence_score":0-1}.\n\n'
        f"LOGS:\n{logs[:12000]}"
    )
    resp = client.chat.completions.create(
        model=s.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{[\s\S]*\}\s*$", content)
    if not m:
        raise ValueError("OpenAI did not return JSON")
    data = json.loads(m.group(0))
    return AnalysisResult(
        root_cause_summary=str(data.get("root_cause_summary", "Unknown")),
        likely_file_or_component=str(data.get("likely_file_or_component", "unknown")),
        suggested_fix=str(data.get("suggested_fix", "")),
        severity=str(data.get("severity", "medium")).lower(),
        confidence_score=float(data.get("confidence_score", 0.5)),
    )


def analyze_failure_logs(logs: str) -> AnalysisResult:
    if get_settings().openai_api_key and OpenAI is not None:
        try:
            return _openai_analyze(logs)
        except Exception:  # noqa: S110
            pass
    return MockAnalyzer().analyze(logs)
