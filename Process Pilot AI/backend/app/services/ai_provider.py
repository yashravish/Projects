import json
import logging
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)

URGENCY_LABELS = {1: "low", 2: "moderate", 3: "medium", 4: "high", 5: "critical"}
IMPACT_LABELS = {1: "minimal", 2: "low", 3: "moderate", 4: "significant", 5: "critical"}


class BaseAIProvider(ABC):
    @abstractmethod
    def generate_summary(
        self,
        title: str,
        description: str,
        category: str,
        urgency: int,
        business_impact: int,
    ) -> dict:
        pass


class OpenAIProvider(BaseAIProvider):
    def __init__(self):
        import openai

        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def generate_summary(
        self,
        title: str,
        description: str,
        category: str,
        urgency: int,
        business_impact: int,
    ) -> dict:
        prompt = (
            "You are a senior business process analyst. Analyze the following business "
            "process request and provide a structured JSON response.\n\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Category: {category}\n"
            f"Urgency (1-5): {urgency}\n"
            f"Business Impact (1-5): {business_impact}\n\n"
            "Respond with ONLY valid JSON containing these keys:\n"
            '- "summary": A concise 2-3 sentence summary of the request and its implications.\n'
            '- "business_impact_explanation": Explain the business impact in concrete terms.\n'
            '- "recommended_action": Specific actionable steps to address this request.\n'
            '- "leadership_summary": A brief executive summary suitable for leadership review.\n'
            '- "implementation_notes": Technical or procedural notes for the implementation team.\n'
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a business process analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except Exception as exc:
            logger.warning("OpenAI call failed (%s), falling back to mock provider", exc)
            return MockAIProvider().generate_summary(
                title, description, category, urgency, business_impact
            )


class MockAIProvider(BaseAIProvider):
    _CATEGORY_TEMPLATES = {
        "access_request": {
            "summary": (
                "This request involves provisioning system access for team members. "
                "Proper access management is essential for operational continuity and "
                "security compliance."
            ),
            "business_impact_explanation": (
                "Delayed access provisioning directly impacts employee productivity "
                "and can create bottlenecks in onboarding workflows. Each day of "
                "delayed access represents lost capacity for the requesting team."
            ),
            "recommended_action": (
                "Verify the access requirements against the organization's role-based "
                "access control policies. Coordinate with IT Security for approval and "
                "provision access through the standard IAM workflow."
            ),
            "leadership_summary": (
                "An access provisioning request has been submitted that affects team "
                "productivity. Recommend expedited processing to minimize operational disruption."
            ),
            "implementation_notes": (
                "Ensure access is granted following the principle of least privilege. "
                "Document all permissions in the access management system for audit purposes."
            ),
        },
        "workflow_issue": {
            "summary": (
                "A workflow inefficiency has been identified that is impacting process "
                "throughput. This issue requires analysis of the current process flow "
                "and identification of optimization opportunities."
            ),
            "business_impact_explanation": (
                "Workflow issues compound over time, leading to increased cycle times "
                "and reduced team capacity. The cumulative effect on operational "
                "efficiency can be substantial if left unaddressed."
            ),
            "recommended_action": (
                "Conduct a process mapping exercise to identify the root cause of the "
                "workflow breakdown. Engage stakeholders to validate findings and "
                "implement targeted process improvements."
            ),
            "leadership_summary": (
                "A workflow issue has been flagged that is affecting process efficiency. "
                "Process improvement analysis is recommended to quantify the impact and "
                "develop a remediation plan."
            ),
            "implementation_notes": (
                "Map the current-state process and identify handoff points where delays "
                "occur. Consider automation opportunities for repetitive manual steps."
            ),
        },
        "data_correction": {
            "summary": (
                "A data quality issue has been reported that requires correction in the "
                "affected system. Data integrity is critical for accurate reporting and "
                "downstream process reliability."
            ),
            "business_impact_explanation": (
                "Inaccurate data can cascade through dependent systems and reports, "
                "leading to flawed decision-making. Timely correction is essential to "
                "maintain data trustworthiness across the organization."
            ),
            "recommended_action": (
                "Validate the scope of the data discrepancy and perform the necessary "
                "corrections. Implement data validation rules to prevent recurrence "
                "and notify downstream consumers of the correction."
            ),
            "leadership_summary": (
                "A data correction request has been submitted to address inaccuracies "
                "in a business-critical system. Prompt resolution will safeguard "
                "reporting accuracy and compliance."
            ),
            "implementation_notes": (
                "Create a backup before making corrections. Log all changes for audit "
                "trail purposes and verify data integrity post-correction."
            ),
        },
        "report_request": {
            "summary": (
                "A new reporting requirement has been identified to support business "
                "analysis and decision-making. The requested report will provide "
                "visibility into key operational metrics."
            ),
            "business_impact_explanation": (
                "Without the requested reporting capability, stakeholders lack the "
                "visibility needed for informed decision-making. This gap can lead to "
                "suboptimal resource allocation and missed opportunities."
            ),
            "recommended_action": (
                "Gather detailed requirements from stakeholders and design the report "
                "structure. Identify data sources, build the report, and establish a "
                "recurring delivery schedule if applicable."
            ),
            "leadership_summary": (
                "A reporting request has been submitted to enhance visibility into "
                "business operations. Fulfilling this request will support data-driven "
                "decision-making for the requesting team."
            ),
            "implementation_notes": (
                "Identify optimal data sources and ensure refresh schedules align with "
                "reporting needs. Consider self-service options to reduce recurring effort."
            ),
        },
        "automation_idea": {
            "summary": (
                "An automation opportunity has been identified that could reduce manual "
                "effort and improve process consistency. Evaluating this idea could yield "
                "significant efficiency gains."
            ),
            "business_impact_explanation": (
                "Manual processes are prone to errors and consume valuable team capacity. "
                "Automation can reduce processing time, improve accuracy, and free up "
                "resources for higher-value activities."
            ),
            "recommended_action": (
                "Assess the feasibility of automation by documenting the current manual "
                "process. Estimate ROI based on time savings and error reduction, then "
                "prioritize against other automation initiatives."
            ),
            "leadership_summary": (
                "An automation opportunity has been proposed that could improve operational "
                "efficiency. A feasibility assessment is recommended to quantify potential "
                "ROI and implementation effort."
            ),
            "implementation_notes": (
                "Document the end-to-end process including exception handling paths. "
                "Evaluate RPA, scripting, and native platform automation options."
            ),
        },
        "process_bottleneck": {
            "summary": (
                "A process bottleneck has been identified that is constraining throughput "
                "and causing delays. This bottleneck requires immediate attention to "
                "restore normal process flow."
            ),
            "business_impact_explanation": (
                "Process bottlenecks create a ripple effect across dependent workflows, "
                "leading to missed deadlines, increased costs, and reduced customer "
                "satisfaction. The longer it persists, the greater the cumulative impact."
            ),
            "recommended_action": (
                "Perform a constraint analysis to isolate the bottleneck root cause. "
                "Implement short-term workarounds to restore flow while developing a "
                "permanent solution. Engage cross-functional stakeholders as needed."
            ),
            "leadership_summary": (
                "A critical process bottleneck has been reported that is impacting "
                "operational throughput. Immediate attention is recommended to minimize "
                "business disruption and restore process efficiency."
            ),
            "implementation_notes": (
                "Use process mining or value stream mapping to quantify the bottleneck. "
                "Consider resource reallocation, parallel processing, or process redesign."
            ),
        },
    }

    _DEFAULT_TEMPLATE = {
        "summary": (
            "A business process request has been submitted that requires analysis "
            "and action. The request should be evaluated based on its urgency and "
            "potential business impact."
        ),
        "business_impact_explanation": (
            "This request has implications for operational efficiency and team "
            "productivity. Timely resolution will help maintain normal business operations."
        ),
        "recommended_action": (
            "Review the request details, assess resource requirements, and develop "
            "an action plan. Coordinate with relevant stakeholders to ensure alignment."
        ),
        "leadership_summary": (
            "A business process request has been submitted for review. The team will "
            "assess the request and provide a resolution plan."
        ),
        "implementation_notes": (
            "Gather additional context from the requester if needed. Document the "
            "resolution approach and any process changes for future reference."
        ),
    }

    def generate_summary(
        self,
        title: str,
        description: str,
        category: str,
        urgency: int,
        business_impact: int,
    ) -> dict:
        template = dict(self._CATEGORY_TEMPLATES.get(category, self._DEFAULT_TEMPLATE))

        urgency_label = URGENCY_LABELS.get(urgency, "medium")
        impact_label = IMPACT_LABELS.get(business_impact, "moderate")

        if urgency >= 4 or business_impact >= 4:
            template["summary"] = (
                f"[HIGH PRIORITY] {template['summary']} Given the {urgency_label} urgency "
                f"and {impact_label} business impact, expedited handling is recommended."
            )
            template["leadership_summary"] = (
                f"{template['leadership_summary']} This item is flagged as high priority "
                f"due to its {urgency_label} urgency level and {impact_label} business impact."
            )
        elif urgency <= 2 and business_impact <= 2:
            template["summary"] = (
                f"{template['summary']} With {urgency_label} urgency and {impact_label} "
                f"impact, this can be addressed through standard processing channels."
            )

        return template


def get_ai_provider() -> BaseAIProvider:
    if settings.ai_provider == "openai" or (
        settings.ai_provider == "auto" and settings.openai_api_key
    ):
        return OpenAIProvider()
    return MockAIProvider()
