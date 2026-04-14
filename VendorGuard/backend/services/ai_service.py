"""
Optional AI integration for generating executive-language summaries.

Controlled by AI_ENABLED feature flag. The application works fully without AI.
When enabled, AI is used ONLY for summarization; all risk scoring is deterministic.
"""

import structlog
from backend.config import get_settings

logger = structlog.get_logger()


def is_ai_enabled() -> bool:
    settings = get_settings()
    return settings.ai_enabled and bool(settings.openai_api_key)


def generate_executive_summary(
    vendor_name: str,
    category: str,
    findings_summary: list[dict],
    inherent_risk: str,
    score: float,
) -> str | None:
    """
    Generate an executive-language risk summary using OpenAI.
    Returns None if AI is disabled or on error.
    """
    if not is_ai_enabled():
        return None

    try:
        from openai import OpenAI
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        findings_text = "\n".join(
            f"- [{f['severity']}] {f['title']}" for f in findings_summary
        )

        prompt = (
            f"You are an information security analyst writing a concise executive summary "
            f"for a third-party security assessment.\n\n"
            f"Vendor: {vendor_name} ({category})\n"
            f"Overall Risk Rating: {inherent_risk} (Score: {score}/100)\n\n"
            f"Key Findings:\n{findings_text}\n\n"
            f"Write a 3-4 sentence executive summary that is formal, concise, "
            f"risk-oriented, and non-alarmist. Focus on the business impact and "
            f"key areas of concern."
        )

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("ai_summary_failed", error=str(e))
        return None


def generate_remediation_narrative(findings: list[dict]) -> str | None:
    """Generate a narrative remediation plan summary."""
    if not is_ai_enabled():
        return None

    try:
        from openai import OpenAI
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        items = "\n".join(
            f"- [{f['severity']}] {f['title']}: {f.get('recommendation', '')}"
            for f in findings
        )

        prompt = (
            f"Based on these security assessment findings and recommendations, "
            f"write a concise remediation plan narrative (4-6 sentences) that "
            f"prioritizes actions by risk level:\n\n{items}"
        )

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("ai_remediation_narrative_failed", error=str(e))
        return None
