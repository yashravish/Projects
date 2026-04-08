"""Tests for evaluator response format and scoring validation."""

import pytest
from pydantic import ValidationError
from app.schemas.evaluation import EvaluationOutput
from app.schemas.checkin import CoachOutput
from app.agents.coach_agent import _contains_crisis_language, SAFE_FALLBACK


class TestEvaluationOutputFormat:
    """Verify the EvaluationOutput schema validation."""

    def test_valid_evaluation(self):
        """Valid scores and notes should create a valid EvaluationOutput."""
        evaluation = EvaluationOutput(
            actionability=8,
            empathy=9,
            specificity=7,
            safety=10,
            overall_notes="Good coaching response.",
        )
        assert evaluation.actionability == 8
        assert evaluation.safety == 10

    def test_scores_must_be_in_range(self):
        """Scores outside 1-10 range should fail validation."""
        with pytest.raises(ValidationError):
            EvaluationOutput(
                actionability=0,  # Below minimum
                empathy=9,
                specificity=7,
                safety=10,
                overall_notes="Invalid score.",
            )

        with pytest.raises(ValidationError):
            EvaluationOutput(
                actionability=8,
                empathy=11,  # Above maximum
                specificity=7,
                safety=10,
                overall_notes="Invalid score.",
            )

    def test_missing_fields_fail(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            EvaluationOutput(
                actionability=8,
                empathy=9,
                # missing specificity, safety, overall_notes
            )

    def test_notes_must_be_string(self):
        """overall_notes must be a string."""
        evaluation = EvaluationOutput(
            actionability=8,
            empathy=9,
            specificity=7,
            safety=10,
            overall_notes="Test notes here.",
        )
        assert isinstance(evaluation.overall_notes, str)


class TestCoachOutputFormat:
    """Verify the CoachOutput schema validation."""

    def test_valid_coach_output(self):
        """Valid coaching fields should create a valid CoachOutput."""
        output = CoachOutput(
            summary="Good progress today.",
            habit_risk="Skipping meals is risky.",
            next_action="Try meal prepping.",
            motivational_message="Keep going!",
        )
        assert output.summary == "Good progress today."

    def test_missing_fields_fail(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            CoachOutput(summary="Only summary provided.")


class TestCrisisDetection:
    """Verify crisis language detection logic."""

    def test_detects_crisis_keywords(self):
        """Should detect known crisis phrases."""
        assert _contains_crisis_language("I want to kill myself") is True
        assert _contains_crisis_language("thinking about self-harm") is True

    def test_passes_safe_input(self):
        """Should not flag normal health inputs."""
        assert _contains_crisis_language("I want to eat healthier") is False
        assert _contains_crisis_language("stressed about work") is False

    def test_safe_fallback_is_valid(self):
        """The safe fallback response should be a valid CoachOutput."""
        assert isinstance(SAFE_FALLBACK, CoachOutput)
        assert "988" in SAFE_FALLBACK.next_action
