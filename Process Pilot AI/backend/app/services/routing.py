from app.services.priority import calculate_priority

CATEGORY_TEAM_MAP = {
    "access_request": "IT_Support",
    "workflow_issue": "Process_Improvement",
    "data_correction": "Data_Team",
    "report_request": "Reports_Analytics",
    "automation_idea": "Automation",
    "process_bottleneck": "Process_Improvement",
}


def route_request(request_obj) -> dict:
    team = CATEGORY_TEAM_MAP.get(request_obj.category, "General")
    priority = calculate_priority(
        request_obj.urgency,
        request_obj.business_impact,
        request_obj.category,
        request_obj.description,
        request_obj.desired_completion_date,
    )
    explanation = _build_explanation(request_obj, team, priority)
    return {
        "suggested_team": team,
        "priority_score": priority,
        "routing_explanation": explanation,
        "category_match": request_obj.category,
    }


def _build_explanation(req, team, priority) -> str:
    urgency_label = {1: "low", 2: "moderate", 3: "medium", 4: "high", 5: "critical"}.get(
        req.urgency, "medium"
    )
    impact_label = {
        1: "minimal",
        2: "low",
        3: "moderate",
        4: "significant",
        5: "critical",
    }.get(req.business_impact, "moderate")
    return (
        f"This {req.category.replace('_', ' ')} request has been assigned to {team} "
        f"based on its category classification. With {urgency_label} urgency and "
        f"{impact_label} business impact, it received a priority score of {priority}/10. "
        f"The request will be reviewed by the {team.replace('_', ' ')} team for appropriate action."
    )
