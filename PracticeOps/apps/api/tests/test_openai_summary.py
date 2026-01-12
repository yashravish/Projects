"""Unit tests for OpenAI summary helpers."""

from app.services.openai_summary import build_fallback_summary, build_summary_prompt


def test_build_fallback_summary_includes_sections_and_overall() -> None:
    sections = [
        {
            "section": "Soprano",
            "member_count": 4,
            "total_practice_days_7d": 16,
            "avg_practice_days_7d": 4.0,
        },
        {
            "section": "Tenor",
            "member_count": 2,
            "total_practice_days_7d": 2,
            "avg_practice_days_7d": 1.0,
        },
    ]

    summary = build_fallback_summary(sections)

    assert "Soprano" in summary
    assert "Tenor" in summary
    assert "Overall average" in summary


def test_build_summary_prompt_contains_json_payload() -> None:
    sections = [
        {
            "section": "Bass",
            "member_count": 3,
            "total_practice_days_7d": 9,
            "avg_practice_days_7d": 3.0,
        }
    ]

    prompt = build_summary_prompt(sections)

    assert "Bass" in prompt
    assert "avg_practice_days_7d" in prompt
