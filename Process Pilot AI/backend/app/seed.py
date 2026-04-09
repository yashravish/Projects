"""Seed the database with demo data. Run with: python -m app.seed"""

import logging
from datetime import date, timedelta

from app.auth.jwt_handler import get_password_hash
from app.database import Base, SessionLocal, engine
from app.models.ai_summary import AISummary
from app.models.request import Request
from app.models.request_update import RequestUpdate
from app.models.routing_decision import RoutingDecision
from app.models.user import User
from app.services.ai_provider import MockAIProvider
from app.services.routing import route_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


USERS = [
    {
        "email": "admin@acme.com",
        "password": "demo123",
        "full_name": "Sarah Chen",
        "department": "Operations",
        "role": "manager",
    },
    {
        "email": "jsmith@acme.com",
        "password": "demo123",
        "full_name": "John Smith",
        "department": "Finance",
        "role": "employee",
    },
    {
        "email": "mgarcia@acme.com",
        "password": "demo123",
        "full_name": "Maria Garcia",
        "department": "HR",
        "role": "employee",
    },
    {
        "email": "dkim@acme.com",
        "password": "demo123",
        "full_name": "David Kim",
        "department": "IT",
        "role": "manager",
    },
    {
        "email": "ljones@acme.com",
        "password": "demo123",
        "full_name": "Lisa Jones",
        "department": "Marketing",
        "role": "employee",
    },
]

REQUESTS_DATA = [
    {
        "title": "SAP System Access for New Finance Team Members",
        "description": "Three new analysts joined the Finance team last week and still lack SAP access. They cannot run reports or process invoices, causing a backlog in accounts payable.",
        "category": "access_request",
        "urgency": 4,
        "business_impact": 4,
        "desired_completion_date": date.today() + timedelta(days=3),
        "user_idx": 1,
        "status": "in_progress",
    },
    {
        "title": "Quarterly Report Generation Taking Too Long",
        "description": "The quarterly financial report now takes over 6 hours to generate due to increased data volume. The finance team needs this optimized before the next quarter-end close.",
        "category": "report_request",
        "urgency": 3,
        "business_impact": 4,
        "desired_completion_date": date.today() + timedelta(days=14),
        "user_idx": 1,
        "status": "submitted",
    },
    {
        "title": "Customer Data Correction in CRM System",
        "description": "A bulk import last month introduced duplicate records and incorrect phone numbers for approximately 500 customer accounts. Sales reps are calling wrong numbers and losing deals.",
        "category": "data_correction",
        "urgency": 5,
        "business_impact": 5,
        "desired_completion_date": date.today() + timedelta(days=5),
        "user_idx": 3,
        "status": "in_review",
    },
    {
        "title": "Automate Monthly Compliance Reporting",
        "description": "The compliance team currently spends 3 days each month manually compiling regulatory reports from multiple systems. An automated solution could free up significant capacity.",
        "category": "automation_idea",
        "urgency": 2,
        "business_impact": 3,
        "desired_completion_date": None,
        "user_idx": 0,
        "status": "submitted",
    },
    {
        "title": "Procurement Approval Workflow Bottleneck",
        "description": "Purchase orders above $5,000 require VP approval, but the current process involves email chains that frequently stall. Average approval time has increased from 2 days to 8 days.",
        "category": "process_bottleneck",
        "urgency": 4,
        "business_impact": 5,
        "desired_completion_date": date.today() + timedelta(days=7),
        "user_idx": 0,
        "status": "in_progress",
    },
    {
        "title": "VPN Access for Remote Marketing Contractors",
        "description": "Four marketing contractors starting next Monday need VPN access to the internal creative assets server. Without access they cannot begin the Q2 campaign work.",
        "category": "access_request",
        "urgency": 3,
        "business_impact": 3,
        "desired_completion_date": date.today() + timedelta(days=5),
        "user_idx": 4,
        "status": "submitted",
    },
    {
        "title": "Employee Onboarding Workflow Needs Streamlining",
        "description": "New hires currently go through 12 separate manual steps across 4 different systems during onboarding. HR spends an average of 4 hours per new hire on administrative setup.",
        "category": "workflow_issue",
        "urgency": 3,
        "business_impact": 4,
        "desired_completion_date": None,
        "user_idx": 2,
        "status": "submitted",
    },
    {
        "title": "Inventory Data Discrepancy Between Warehouse and ERP",
        "description": "Physical inventory counts show a 15% variance from ERP records for the eastern warehouse. This is causing fulfillment delays and customer complaints.",
        "category": "data_correction",
        "urgency": 4,
        "business_impact": 4,
        "desired_completion_date": date.today() + timedelta(days=7),
        "user_idx": 0,
        "status": "in_review",
    },
    {
        "title": "Automate Invoice Matching Process",
        "description": "Accounts payable manually matches 200+ invoices per week against purchase orders and receiving documents. Automating the three-way match could save 20 hours per week.",
        "category": "automation_idea",
        "urgency": 2,
        "business_impact": 4,
        "desired_completion_date": None,
        "user_idx": 1,
        "status": "submitted",
    },
    {
        "title": "IT Helpdesk Ticket Resolution Bottleneck",
        "description": "Average ticket resolution time has increased from 4 hours to 2 days due to understaffing and lack of a proper triage system. Critical issues are not being prioritized.",
        "category": "process_bottleneck",
        "urgency": 5,
        "business_impact": 4,
        "desired_completion_date": date.today() + timedelta(days=3),
        "user_idx": 3,
        "status": "in_progress",
    },
    {
        "title": "Weekly Sales Pipeline Dashboard Request",
        "description": "Sales leadership needs a real-time dashboard showing pipeline stages, conversion rates, and forecast accuracy. Currently this data is compiled manually in spreadsheets each Friday.",
        "category": "report_request",
        "urgency": 3,
        "business_impact": 3,
        "desired_completion_date": date.today() + timedelta(days=21),
        "user_idx": 4,
        "status": "submitted",
    },
    {
        "title": "Budget Approval Workflow Stuck in Legacy System",
        "description": "The annual budget approval process still runs through a 10-year-old SharePoint workflow that frequently errors out. Department heads are unable to submit revisions on time.",
        "category": "workflow_issue",
        "urgency": 4,
        "business_impact": 5,
        "desired_completion_date": date.today() + timedelta(days=10),
        "user_idx": 1,
        "status": "pending_info",
    },
    {
        "title": "Grant Database Access to External Auditors",
        "description": "External auditors arriving next week need read-only access to the financial databases for the annual audit. This requires IT security review and temporary credential provisioning.",
        "category": "access_request",
        "urgency": 5,
        "business_impact": 3,
        "desired_completion_date": date.today() + timedelta(days=2),
        "user_idx": 1,
        "status": "resolved",
    },
    {
        "title": "Automate Employee Offboarding Checklist",
        "description": "When employees leave, 8 different teams must be notified and access revoked from 12 systems. This manual process has led to security gaps where former employees retained access.",
        "category": "automation_idea",
        "urgency": 3,
        "business_impact": 4,
        "desired_completion_date": None,
        "user_idx": 2,
        "status": "submitted",
    },
    {
        "title": "Cross-Department Data Sharing Bottleneck",
        "description": "Marketing, Sales, and Product teams all maintain separate customer data stores. Reconciling this data for campaigns takes a full week each month and results are often inconsistent.",
        "category": "process_bottleneck",
        "urgency": 3,
        "business_impact": 4,
        "desired_completion_date": None,
        "user_idx": 4,
        "status": "submitted",
    },
    {
        "title": "Correct Payroll Tax Withholding Errors for Q1",
        "description": "Twelve employees in the California office had incorrect state tax withholding for Q1 due to a configuration error after the January system update. Corrections need to be applied retroactively.",
        "category": "data_correction",
        "urgency": 5,
        "business_impact": 4,
        "desired_completion_date": date.today() + timedelta(days=5),
        "user_idx": 2,
        "status": "in_progress",
    },
    {
        "title": "Monthly Department KPI Report Automation",
        "description": "Each department head manually compiles their KPI report from 3 different data sources. An automated report with a standard template would save approximately 5 hours per department per month.",
        "category": "report_request",
        "urgency": 2,
        "business_impact": 3,
        "desired_completion_date": None,
        "user_idx": 0,
        "status": "closed",
    },
    {
        "title": "Vendor Onboarding Process Workflow Redesign",
        "description": "New vendor setup currently requires 15 steps across procurement, legal, and finance. The average time from vendor selection to first PO is 6 weeks, which is unacceptable for urgent needs.",
        "category": "workflow_issue",
        "urgency": 3,
        "business_impact": 3,
        "desired_completion_date": date.today() + timedelta(days=30),
        "user_idx": 0,
        "status": "submitted",
    },
]


def seed():
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing_users = db.query(User).count()
        if existing_users > 0:
            logger.info("Database already seeded (%d users found). Skipping.", existing_users)
            return

        logger.info("Seeding users...")
        users = []
        for u in USERS:
            user = User(
                email=u["email"],
                full_name=u["full_name"],
                department=u["department"],
                role=u["role"],
                hashed_password=get_password_hash(u["password"]),
            )
            db.add(user)
            users.append(user)
        db.flush()
        logger.info("Created %d users.", len(users))

        logger.info("Seeding requests...")
        ai_provider = MockAIProvider()
        requests_objs = []
        for i, rd in enumerate(REQUESTS_DATA):
            req = Request(
                requester_id=users[rd["user_idx"]].id,
                title=rd["title"],
                description=rd["description"],
                category=rd["category"],
                urgency=rd["urgency"],
                business_impact=rd["business_impact"],
                desired_completion_date=rd.get("desired_completion_date"),
                status=rd.get("status", "submitted"),
            )
            db.add(req)
            db.flush()

            routing_result = route_request(req)
            routing = RoutingDecision(
                request_id=req.id,
                suggested_team=routing_result["suggested_team"],
                priority_score=routing_result["priority_score"],
                routing_explanation=routing_result["routing_explanation"],
                category_match=routing_result["category_match"],
            )
            db.add(routing)

            req.priority_score = routing_result["priority_score"]
            req.assigned_team = routing_result["suggested_team"]

            requests_objs.append(req)
            logger.info("  [%d/%d] Created request: %s", i + 1, len(REQUESTS_DATA), rd["title"][:50])

        db.flush()

        logger.info("Adding request updates...")
        update_data = [
            (0, 3, "in_progress", "Submitted access request to IT Security team. Awaiting approval."),
            (0, 3, None, "IT Security approved access. Provisioning in SAP now."),
            (2, 3, "in_review", "Identified the scope of duplicate records. Planning correction batch."),
            (4, 0, "in_progress", "Initiated workflow analysis with procurement team."),
            (4, 0, None, "Identified three key approval bottlenecks. Drafting process improvement proposal."),
            (7, 3, "in_review", "Comparing physical counts with ERP data to identify discrepancy sources."),
            (9, 3, "in_progress", "Implementing ticket triage system. Reassigning resources."),
            (9, 3, None, "New triage process reduced average response time by 40%."),
            (11, 0, "pending_info", "Waiting for department heads to confirm budget revision requirements."),
            (12, 3, "resolved", "Temporary credentials provisioned. Auditors confirmed access is working."),
            (15, 0, "in_progress", "Payroll team is running correction scripts for affected employees."),
            (16, 0, "closed", "Monthly KPI reports are now automated. Template deployed to all departments."),
        ]
        for req_idx, author_idx, status_change, note in update_data:
            update = RequestUpdate(
                request_id=requests_objs[req_idx].id,
                author_id=users[author_idx].id,
                status_change=status_change,
                note=note,
            )
            db.add(update)

        logger.info("Generating AI summaries for select requests...")
        summary_indices = [0, 2, 4, 7, 9, 14]
        for idx in summary_indices:
            req = requests_objs[idx]
            result = ai_provider.generate_summary(
                title=req.title,
                description=req.description,
                category=req.category,
                urgency=req.urgency,
                business_impact=req.business_impact,
            )
            summary = AISummary(
                request_id=req.id,
                summary=result["summary"],
                business_impact_explanation=result["business_impact_explanation"],
                recommended_action=result["recommended_action"],
                leadership_summary=result["leadership_summary"],
                implementation_notes=result.get("implementation_notes"),
                provider_used="MockAIProvider",
            )
            db.add(summary)
            logger.info("  Generated AI summary for: %s", req.title[:50])

        db.commit()
        logger.info("Seeding complete! Created %d users and %d requests.", len(users), len(requests_objs))

    except Exception:
        db.rollback()
        logger.exception("Seeding failed")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
