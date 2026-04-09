from types import SimpleNamespace

from app.services.routing import CATEGORY_TEAM_MAP, route_request


def test_category_team_mapping():
    assert CATEGORY_TEAM_MAP["access_request"] == "IT_Support"
    assert CATEGORY_TEAM_MAP["workflow_issue"] == "Process_Improvement"
    assert CATEGORY_TEAM_MAP["data_correction"] == "Data_Team"
    assert CATEGORY_TEAM_MAP["report_request"] == "Reports_Analytics"
    assert CATEGORY_TEAM_MAP["automation_idea"] == "Automation"
    assert CATEGORY_TEAM_MAP["process_bottleneck"] == "Process_Improvement"


def test_route_request_returns_all_keys():
    mock_req = SimpleNamespace(
        category="access_request",
        urgency=3,
        business_impact=3,
        description="Need access to the reporting system.",
        desired_completion_date=None,
    )
    result = route_request(mock_req)
    assert "suggested_team" in result
    assert "priority_score" in result
    assert "routing_explanation" in result
    assert "category_match" in result
    assert result["suggested_team"] == "IT_Support"
    assert result["category_match"] == "access_request"
    assert isinstance(result["priority_score"], float)
    assert isinstance(result["routing_explanation"], str)
