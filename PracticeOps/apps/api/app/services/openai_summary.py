"""OpenAI summary generation helpers."""

from __future__ import annotations

import json
from enum import Enum

import httpx

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SummarySource(str, Enum):
    """Summary generation source."""

    OPENAI = "openai"
    FALLBACK = "fallback"


class OpenAISummaryError(RuntimeError):
    """Raised when OpenAI summarization fails."""


def build_fallback_summary(sections: list[dict[str, float | int | str]]) -> str:
    """Build a deterministic summary from section stats."""
    if not sections:
        return "No compliance data available for the last 7 days."

    sorted_sections = sorted(
        sections, key=lambda s: float(s["avg_practice_days_7d"]), reverse=True
    )
    top = sorted_sections[0]
    bottom = sorted_sections[-1]

    total_members = sum(int(s["member_count"]) for s in sections)
    total_days = sum(int(s["total_practice_days_7d"]) for s in sections)
    overall_avg = round(total_days / total_members, 1) if total_members else 0.0

    return (
        "Practice compliance varies by section. "
        f"Top section: {top['section']} ({float(top['avg_practice_days_7d']):.1f} days avg). "
        f"Lowest: {bottom['section']} ({float(bottom['avg_practice_days_7d']):.1f} days avg). "
        f"Overall average: {overall_avg:.1f} days per member."
    )


def build_summary_prompt(sections: list[dict[str, float | int | str]]) -> str:
    """Build a compact prompt for OpenAI."""
    return (
        "Summarize the compliance chart in 1-2 sentences. "
        "Call out the top and bottom sections and the overall average. "
        "Use concise, neutral language. Data:\n"
        f"{json.dumps(sections, indent=2)}"
    )


async def _request_openai_summary(prompt: str) -> str:
    """Call OpenAI Chat Completions API and return the summary."""
    if not settings.openai_api_key:
        raise OpenAISummaryError("OpenAI API key not configured")

    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that summarizes analytics charts for team leaders."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    async with httpx.AsyncClient(base_url=settings.openai_base_url, timeout=10.0) as client:
        response = await client.post("/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise OpenAISummaryError("Unexpected OpenAI response format") from exc


async def generate_compliance_summary(
    sections: list[dict[str, float | int | str]]
) -> tuple[str, SummarySource]:
    """Generate compliance summary using OpenAI with a fallback."""
    if not sections:
        return "No compliance data available for the last 7 days.", SummarySource.FALLBACK

    prompt = build_summary_prompt(sections)
    try:
        summary = await _request_openai_summary(prompt)
        return summary, SummarySource.OPENAI
    except (OpenAISummaryError, httpx.HTTPError) as exc:
        logger.warning("openai_summary_failed", error=str(exc))
        return build_fallback_summary(sections), SummarySource.FALLBACK
