import pytest

from app.models.pipeline import StageName
from app.services.ai_analyzer import MockAnalyzer, analyze_failure_logs
from app.services.feature_flags import is_flag_granted
from app.services.hash_util import stable_user_bucket, stable_variant_choice
from app.services.pipeline_service import _should_force_failure


def test_variant_deterministic():
    assert stable_variant_choice("u1", 50) == stable_variant_choice("u1", 50)
    assert stable_variant_choice("u1", 0) == "B"
    assert stable_variant_choice("u1", 100) == "A"


def test_user_bucket_stable():
    assert stable_user_bucket("abc", 100) == stable_user_bucket("abc", 100)
    assert 0 <= stable_user_bucket("x", 100) < 100


def test_feature_flag_rollout():
    assert is_flag_granted(100, "u", 1, True) is True
    assert is_flag_granted(0, "u", 1, True) is False
    assert is_flag_granted(50, "u", 1, False) is False


def test_mock_analyzer_patterns():
    m = MockAnalyzer()
    r = m.analyze("ModuleNotFoundError: No module named 'pandas'")
    assert "package" in r.root_cause_summary.lower() or "package" in r.suggested_fix.lower()
    assert r.confidence_score > 0.5


def test_analyze_fallback_mock(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app.core.config import get_settings

    get_settings.cache_clear()
    r = analyze_failure_logs("ECONNREFUSED connecting to redis:6379")
    assert "connection" in r.root_cause_summary.lower() or "connection" in r.suggested_fix.lower()


def test_pipeline_failure_determinism():
    st = _should_force_failure(7, 1)
    assert st.fail_at_stage == StageName.unit_tests
    st2 = _should_force_failure(7, 2)
    assert st2.fail_at_stage == StageName.unit_tests
