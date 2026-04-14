"""
Transparent, weighted risk scoring model.

Score Calculation
-----------------
1. Each finding has a severity: Critical=4, High=3, Moderate=2, Low=1.
2. Each control domain has an importance weight (1-5).
3. Inherent risk points = sum of (severity_value * domain_weight) for all findings.
4. Maximum possible points = number_of_rules * max_severity(4) * max_weight(5).
   To keep scores practical, we cap the denominator at a fixed baseline of 200.
5. Inherent risk score = (total_points / baseline) * 100, capped at 100.
6. Residual risk score = inherent - remediation_credit (based on mitigated findings).

Risk Rating Thresholds
----------------------
- 0-25:  Low
- 26-50: Moderate
- 51-75: High
- 76-100: Critical
"""

SEVERITY_VALUES = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4,
}

DOMAIN_WEIGHTS = {
    "AC": 5,   # Access Control
    "DP": 5,   # Data Protection
    "AM": 3,   # Asset Management
    "VM": 3,   # Vendor Management
    "LM": 4,   # Logging and Monitoring
    "IR": 4,   # Incident Response
    "VU": 4,   # Vulnerability Management
    "SC": 3,   # Secure Configuration
    "BC": 3,   # Business Continuity
    "GD": 3,   # Governance and Documentation
    "AG": 4,   # AI Governance
    "IOT": 4,  # IoT Security
}

SCORE_BASELINE = 200

RATING_THRESHOLDS = [
    (25, "Low"),
    (50, "Moderate"),
    (75, "High"),
    (100, "Critical"),
]


def severity_to_value(severity: str) -> int:
    return SEVERITY_VALUES.get(severity, 1)


def domain_weight(domain_code: str) -> int:
    return DOMAIN_WEIGHTS.get(domain_code, 3)


def score_to_rating(score: float) -> str:
    for threshold, rating in RATING_THRESHOLDS:
        if score <= threshold:
            return rating
    return "Critical"


def calculate_inherent_risk(findings: list) -> tuple:
    """
    Calculate the inherent risk score from a list of finding dicts.

    Each finding dict must contain 'severity' and 'domain_code'.

    Returns (score: float, rating: str).
    """
    if not findings:
        return (0.0, "Low")

    total_points = 0
    for f in findings:
        sev = severity_to_value(f.get("severity", "Low"))
        weight = domain_weight(f.get("domain_code", "SC"))
        total_points += sev * weight

    score = min((total_points / SCORE_BASELINE) * 100, 100.0)
    score = round(score, 1)
    return (score, score_to_rating(score))


def calculate_residual_risk(inherent_score: float, findings: list) -> tuple:
    """
    Estimate residual risk based on mitigated findings.

    Mitigated or closed findings reduce the inherent score proportionally.

    Returns (score: float, rating: str).
    """
    if not findings or inherent_score == 0:
        return (0.0, "Low")

    total_weight = 0
    mitigated_weight = 0
    for f in findings:
        sev = severity_to_value(f.get("severity", "Low"))
        weight = domain_weight(f.get("domain_code", "SC"))
        pts = sev * weight
        total_weight += pts
        status = f.get("remediation_status", "open")
        if status in ("mitigated", "closed"):
            mitigated_weight += pts
        elif status == "in_progress":
            mitigated_weight += pts * 0.25
        elif status == "accepted_risk":
            mitigated_weight += pts * 0.1

    if total_weight == 0:
        return (0.0, "Low")

    reduction_ratio = mitigated_weight / total_weight
    residual = inherent_score * (1 - reduction_ratio)
    residual = round(max(residual, 0), 1)
    return (residual, score_to_rating(residual))
