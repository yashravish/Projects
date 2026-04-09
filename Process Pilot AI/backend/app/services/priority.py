from datetime import date, timedelta

CATEGORY_WEIGHTS = {
    "process_bottleneck": 0.5,
    "workflow_issue": 0.3,
    "data_correction": 0.2,
    "access_request": 0.1,
    "automation_idea": 0.0,
    "report_request": -0.2,
}

BOOST_KEYWORDS = [
    "urgent",
    "blocked",
    "critical",
    "deadline",
    "revenue",
    "compliance",
    "security",
    "down",
    "outage",
    "broken",
]


def calculate_priority(
    urgency: int,
    business_impact: int,
    category: str,
    description: str,
    desired_completion_date=None,
) -> float:
    base = (urgency + business_impact) / 2.0 * 2.0
    cat_weight = CATEGORY_WEIGHTS.get(category, 0.0)
    kw_boost = min(
        sum(0.15 for kw in BOOST_KEYWORDS if kw in description.lower()), 1.0
    )
    deadline_boost = 0.0
    if desired_completion_date:
        days_left = (desired_completion_date - date.today()).days
        if days_left <= 3:
            deadline_boost = 1.0
        elif days_left <= 7:
            deadline_boost = 0.5
        elif days_left <= 14:
            deadline_boost = 0.25
    total = base + cat_weight + kw_boost + deadline_boost
    return round(max(1.0, min(10.0, total)), 1)
