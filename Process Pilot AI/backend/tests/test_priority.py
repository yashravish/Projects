from datetime import date, timedelta

from app.services.priority import calculate_priority


def test_base_score_high():
    score = calculate_priority(
        urgency=5,
        business_impact=5,
        category="process_bottleneck",
        description="Normal request description.",
    )
    assert score >= 9.0
    assert score <= 10.0


def test_low_priority():
    score = calculate_priority(
        urgency=1,
        business_impact=1,
        category="report_request",
        description="Just a routine report request.",
    )
    assert score >= 1.0
    assert score <= 3.0


def test_keyword_boost():
    score_no_kw = calculate_priority(
        urgency=3,
        business_impact=3,
        category="workflow_issue",
        description="Normal workflow request.",
    )
    score_with_kw = calculate_priority(
        urgency=3,
        business_impact=3,
        category="workflow_issue",
        description="Critical blocked issue causing outage.",
    )
    assert score_with_kw > score_no_kw


def test_deadline_boost():
    soon = date.today() + timedelta(days=2)
    score_no_deadline = calculate_priority(
        urgency=3,
        business_impact=3,
        category="access_request",
        description="Standard access request.",
    )
    score_with_deadline = calculate_priority(
        urgency=3,
        business_impact=3,
        category="access_request",
        description="Standard access request.",
        desired_completion_date=soon,
    )
    assert score_with_deadline > score_no_deadline


def test_score_clamping():
    low = calculate_priority(
        urgency=1,
        business_impact=1,
        category="report_request",
        description="Nothing special.",
    )
    high = calculate_priority(
        urgency=5,
        business_impact=5,
        category="process_bottleneck",
        description="Urgent critical blocked deadline revenue compliance security down outage broken.",
        desired_completion_date=date.today(),
    )
    assert low >= 1.0
    assert high <= 10.0
