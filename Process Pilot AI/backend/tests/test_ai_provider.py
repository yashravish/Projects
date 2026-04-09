from unittest.mock import patch

from app.services.ai_provider import MockAIProvider, get_ai_provider


def test_mock_provider_returns_all_keys():
    provider = MockAIProvider()
    result = provider.generate_summary(
        title="Test Request",
        description="A test description for the mock provider.",
        category="workflow_issue",
        urgency=3,
        business_impact=3,
    )
    assert "summary" in result
    assert "business_impact_explanation" in result
    assert "recommended_action" in result
    assert "leadership_summary" in result
    assert "implementation_notes" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_get_ai_provider_returns_mock_when_no_key():
    with patch("app.services.ai_provider.settings") as mock_settings:
        mock_settings.ai_provider = "auto"
        mock_settings.openai_api_key = ""
        provider = get_ai_provider()
        assert isinstance(provider, MockAIProvider)


def test_mock_provider_deterministic():
    provider = MockAIProvider()
    result1 = provider.generate_summary(
        title="Same Title",
        description="Same description for determinism test.",
        category="data_correction",
        urgency=4,
        business_impact=4,
    )
    result2 = provider.generate_summary(
        title="Same Title",
        description="Same description for determinism test.",
        category="data_correction",
        urgency=4,
        business_impact=4,
    )
    assert result1 == result2
