# VendorGuard — Enterprise Third-Party Security Assessment Platform

**VendorGuard** is a production-style internal security assessment platform designed for enterprise Information Security teams to evaluate third-party technologies. It supports structured security assessments for SaaS, PaaS, AI tools, IoT platforms, tokenization platforms, distributed ledger technologies, and end-user software packages.

This project demonstrates realistic enterprise security workflows including vendor intake, structured questionnaire-based assessments, deterministic risk scoring, control-domain mapping inspired by NIST CSF and ISO 27001, remediation tracking, and professional report generation.

---

## Why This Project Matters

Enterprise organizations rely on dozens to hundreds of third-party technologies. Each one introduces supply chain risk — from data exposure and access control gaps to compliance blind spots and emerging technology concerns (AI data retention, IoT segmentation, blockchain immutability).

VendorGuard simulates the tools and workflows that Information Security teams use daily to:
- Evaluate new technology requests before deployment
- Document security posture with structured evidence
- Score risk transparently so stakeholders understand the reasoning
- Track remediation to ensure findings are addressed
- Generate audit-ready reports for governance and compliance

---

## Architecture

```
┌──────────────────────────────────────────────┐
│                  Browser (UI)                │
│   Jinja2 Server-Rendered Templates           │
│   + Vanilla JS (fetch → REST API)            │
└──────────────────┬───────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────┐
│              FastAPI Application              │
│  ┌─────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Routers │ │ Services │ │ Risk Engine   │  │
│  │ (API +  │ │ (Audit,  │ │ (Rules,       │  │
│  │  Pages) │ │  AI, PDF)│ │  Scoring,     │  │
│  └────┬────┘ └────┬─────┘ │  Domains)     │  │
│       │           │       └───────────────┘  │
│  ┌────▼───────────▼──────────────────────┐   │
│  │       SQLAlchemy ORM + Models         │   │
│  └───────────────┬───────────────────────┘   │
└──────────────────┼───────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │    PostgreSQL      │
         └───────────────────┘
```

### Tech Stack

| Layer      | Technology                                        |
|------------|---------------------------------------------------|
| Backend    | Python 3.12, FastAPI, SQLAlchemy, Pydantic         |
| Database   | PostgreSQL (Docker) / SQLite (tests)               |
| Frontend   | Jinja2 templates, vanilla HTML/CSS/JS              |
| Reports    | Jinja2 + WeasyPrint (PDF) / HTML fallback          |
| Auth       | JWT (python-jose) with cookie + Bearer support     |
| Migrations | Alembic                                            |
| Testing    | pytest                                             |
| CI         | GitHub Actions                                     |
| Deploy     | Docker + docker-compose                            |
| AI (opt.)  | OpenAI API (feature-flagged, summarization only)   |

---

## Features

### Core Workflows
- **Vendor Intake** — Register third-party technologies with metadata, hosting model, data types, and compliance attestations
- **Assessment Questionnaire** — Structured multi-section questionnaire covering data handling, IAM, encryption, logging, incident response, compliance, and category-specific controls (AI, IoT, tokenization, DLT, end-user software)
- **Deterministic Risk Engine** — 29 transparent rules that evaluate answers and produce findings with severity, likelihood, impact, and remediation recommendations
- **Weighted Risk Scoring** — Inherent and residual risk scores (0–100) using control domain weights and severity values
- **Control Domain Mapping** — 12 domains inspired by NIST CSF and ISO 27001
- **Findings Dashboard** — Filter by severity, status, and control domain
- **Remediation Tracker** — Assign owners, set due dates, track status (open → in progress → mitigated → accepted risk → closed)
- **Report Generation** — Professional PDF/HTML reports with cover page, methodology, risk summary, detailed findings, and remediation plan
- **Assessment Templates** — Reusable questionnaire templates for each vendor category
- **Audit Trail** — All significant actions logged with timestamps and user attribution
- **Dashboard** — KPIs including vendor counts, severity distribution, domain breakdown, and recent activity

### Security & Quality
- JWT authentication with role-based access (admin/analyst)
- Input validation via Pydantic
- Structured logging (structlog)
- Environment variable configuration
- Docker containerization
- GitHub Actions CI pipeline
- pytest test suite (unit + integration)

### Optional AI Integration
- Feature-flagged (`AI_ENABLED=true`)
- Summarization only — risk scoring remains 100% deterministic
- Generates executive-language summaries and remediation narratives
- App works fully without AI

---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- OR: Python 3.12+, PostgreSQL

### Quick Start (Docker)

```bash
# Clone the repository
git clone <repo-url> VendorGuard && cd VendorGuard

# Copy environment file
cp .env.example .env

# Start the application
docker-compose up --build

# The app seeds demo data automatically on first run
# Open http://localhost:8000
```

### Local Development (without Docker)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL and update .env
cp .env.example .env
# Edit DATABASE_URL in .env to point to your PostgreSQL instance

# Run database migrations
alembic upgrade head

# Seed demo data
python -m backend.seed

# Start the application
uvicorn backend.main:app --reload --port 8000
```

### Environment Variables

| Variable                      | Description                       | Default                          |
|-------------------------------|-----------------------------------|----------------------------------|
| `DATABASE_URL`                | PostgreSQL connection string      | (see .env.example)               |
| `SECRET_KEY`                  | JWT signing key                   | change-me-...                    |
| `AI_ENABLED`                  | Enable OpenAI integration         | `false`                          |
| `OPENAI_API_KEY`              | OpenAI API key (if AI enabled)    | (empty)                          |
| `OPENAI_MODEL`                | OpenAI model to use               | `gpt-4`                          |
| `LOG_LEVEL`                   | Logging level                     | `INFO`                           |

### Demo Credentials

| Role    | Username  | Password    |
|---------|-----------|-------------|
| Admin   | admin     | admin123    |
| Analyst | analyst   | analyst123  |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=term-missing

# Run specific test file
pytest tests/test_risk_engine.py -v
```

Tests use an in-memory SQLite database — no PostgreSQL required for testing.

---

## API Endpoints

| Method | Endpoint                            | Description                    |
|--------|-------------------------------------|--------------------------------|
| POST   | `/api/auth/login`                   | Authenticate user              |
| POST   | `/api/auth/logout`                  | Clear session                  |
| GET    | `/api/auth/me`                      | Current user info              |
| GET    | `/api/vendors`                      | List vendors                   |
| POST   | `/api/vendors`                      | Create vendor                  |
| GET    | `/api/vendors/{id}`                 | Vendor detail                  |
| POST   | `/api/vendors/{id}/integrations`    | Add integration                |
| GET    | `/api/assessments`                  | List assessments               |
| POST   | `/api/assessments`                  | Create assessment              |
| GET    | `/api/assessments/{id}`             | Assessment detail              |
| POST   | `/api/assessments/{id}/submit`      | Submit answers                 |
| POST   | `/api/assessments/{id}/evaluate`    | Run risk engine                |
| GET    | `/api/findings`                     | List findings (filterable)     |
| GET    | `/api/remediation`                  | List remediation items         |
| PATCH  | `/api/remediation/{id}`             | Update remediation status      |
| GET    | `/api/reports/{assessment_id}`      | HTML report preview            |
| POST   | `/api/reports/{assessment_id}/generate` | Generate PDF report        |
| GET    | `/api/dashboard`                    | Dashboard statistics           |
| GET    | `/api/templates`                    | Assessment templates           |
| GET    | `/api/templates/domains/list`       | Control domain reference       |
| GET    | `/api/health`                       | Health check                   |

---

## Risk Scoring Model

### How It Works

The risk engine uses a transparent, deterministic weighted scoring model:

1. **Rules Evaluation**: 29 rules check vendor metadata and assessment answers. Each triggered rule produces a finding with a severity (Low=1, Moderate=2, High=3, Critical=4).

2. **Domain Weighting**: Each finding maps to a control domain. Domains have importance weights (1–5):
   - Weight 5: Access Control, Data Protection
   - Weight 4: Logging & Monitoring, Incident Response, Vulnerability Management, AI Governance, IoT Security
   - Weight 3: Asset Management, Vendor Management, Secure Configuration, Business Continuity, Governance & Documentation

3. **Score Calculation**:
   - Inherent Risk Points = Σ (severity_value × domain_weight) for all findings
   - Inherent Risk Score = (points / 200) × 100, capped at 100
   - Risk Rating: 0–25 = Low, 26–50 = Moderate, 51–75 = High, 76–100 = Critical

4. **Residual Risk**: Calculated by reducing the inherent score proportionally based on remediation status:
   - Mitigated/Closed: 100% credit
   - In Progress: 25% credit
   - Accepted Risk: 10% credit

### Example Rules

| Condition | Severity | Domain |
|-----------|----------|--------|
| Sensitive data without encryption at rest | Critical/High | Data Protection |
| No MFA with privileged vendor access | High | Access Control |
| AI tool with broad data access + unclear retention | Critical/High | AI Governance |
| IoT platform without network segmentation | High | IoT Security |
| IoT default credentials not changed | Critical | IoT Security |
| No documented incident response plan | High/Moderate | Incident Response |
| Tokenization without documented key management | High | Data Protection |
| DLT with PII on-chain, no privacy controls | High | Data Protection |
| End-user software: local admin + no auto-update | High | Vulnerability Mgmt |
| No compliance attestations for sensitive data handler | Moderate | Governance |

All rules are defined in `backend/engine/rules.py` and are designed to be readable, explainable, and interview-ready.

---

## Control Domain Reference

| Code | Domain | NIST CSF | ISO 27001 |
|------|--------|----------|-----------|
| AC | Access Control | PR.AC | A.9 |
| DP | Data Protection | PR.DS | A.8, A.10 |
| AM | Asset Management | ID.AM | A.8 |
| VM | Vendor Management | ID.SC | A.15 |
| LM | Logging and Monitoring | DE.CM | A.12.4 |
| IR | Incident Response | RS, RC | A.16 |
| VU | Vulnerability Management | DE.CM, RS.MI | A.12.6 |
| SC | Secure Configuration | PR.IP | A.14 |
| BC | Business Continuity | PR.IP, RC.RP | A.17 |
| GD | Governance and Documentation | ID.GV | A.5, A.18 |
| AG | AI Governance | NIST AI RMF | ISO/IEC 42001 |
| IOT | IoT Security | NISTIR 8259 | ISO/IEC 27400 |

*Mappings are illustrative and educational — not formal compliance advice.*

---

## Seed Data Scenarios

The database is pre-populated with 5 realistic vendor assessments:

| Vendor | Category | Key Findings |
|--------|----------|-------------|
| **PeopleForce HR** | SaaS | Logs not exportable, no DR plan, no right to audit |
| **NoteGenius AI** | AI Tool | Broad data access, trains on customer data, no MFA, no logging, no IR plan, no encryption at rest |
| **OfficeSense IoT** | IoT Platform | No segmentation, no device inventory, default creds unchanged, no compliance certs |
| **VaultPay Tokenization** | Tokenization | Key management not documented, no HSM, excessive integrations |
| **AuditChain DLT** | Distributed Ledger | PII on-chain without privacy controls |

---

## Project Structure

```
VendorGuard/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Settings (pydantic-settings)
│   ├── database.py          # SQLAlchemy engine and session
│   ├── auth.py              # JWT authentication
│   ├── models.py            # SQLAlchemy ORM models (11 tables)
│   ├── schemas.py           # Pydantic request/response models
│   ├── seed.py              # Database seeding script
│   ├── engine/
│   │   ├── domain_mapping.py  # 12 control domains (NIST/ISO)
│   │   ├── questionnaire.py   # Assessment question definitions
│   │   ├── rules.py           # 29 deterministic risk rules
│   │   ├── scoring.py         # Weighted risk scoring model
│   │   └── risk_engine.py     # Orchestrator
│   ├── services/
│   │   ├── ai_service.py      # Optional OpenAI integration
│   │   ├── report_service.py  # PDF/HTML report generation
│   │   └── audit_service.py   # Audit trail logging
│   └── routers/
│       ├── auth.py            # Authentication endpoints
│       ├── vendors.py         # Vendor CRUD
│       ├── assessments.py     # Assessment workflow
│       ├── findings.py        # Findings queries
│       ├── remediation.py     # Remediation tracking
│       ├── reports.py         # Report generation
│       ├── dashboard.py       # Dashboard statistics
│       ├── templates_router.py # Governance templates
│       └── pages.py           # HTML page routes
├── templates/                 # Jinja2 HTML templates
├── static/                    # CSS and JavaScript
├── tests/                     # pytest test suite
├── alembic/                   # Database migrations
├── .github/workflows/ci.yml   # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Limitations

- Authentication is simplified for demo purposes (no password reset, no account management)
- PDF export requires WeasyPrint system dependencies; falls back to HTML if unavailable
- AI integration requires an OpenAI API key and is entirely optional
- Assessment questionnaires are currently not saveable as draft (submit evaluates immediately)
- No file upload for compliance evidence (metadata-only tracking)
- Single-user workflow (no approval chains or multi-reviewer flows)

## Future Improvements

- Evidence attachment upload with file storage
- Multi-reviewer approval workflow
- Vendor tiering (Critical / High / Medium / Low) with SLA-driven review cadence
- Risk heatmap visualization
- CSV export for findings and remediation items
- Assessment comparison (pre vs. post-implementation)
- Policy exception and risk acceptance formal workflow
- Email notifications for due dates and status changes
- SSO integration for production deployment (SAML/OIDC)
- RBAC enhancements (team-based access, vendor ownership)

---

## How This Project Aligns to InfoSec / Supply Chain Security Roles

This project directly demonstrates competencies sought in these roles:

| Competency Area | How VendorGuard Demonstrates It |
|-----------------|--------------------------------|
| **Third-party risk assessment** | Full vendor intake → assessment → findings → remediation workflow |
| **Supply chain security** | Subprocessor documentation tracking, 4th-party awareness |
| **Governance & documentation** | Audit trails, assessment templates, control domain references |
| **Structured risk evaluation** | Transparent, deterministic rules engine with weighted scoring |
| **Security communication** | Professional report generation with executive summaries |
| **Emerging technology review** | Dedicated assessment sections for AI, IoT, tokenization, DLT |
| **Framework awareness** | NIST CSF and ISO 27001 inspired domain mappings |
| **Remediation management** | Owner assignment, due dates, status tracking, residual risk |
| **Auditability** | Complete audit log of all assessment lifecycle actions |
| **Technical implementation** | Clean API design, modular architecture, testing, Docker, CI |

---

## Resume-Ready Project Entry

**VendorGuard — Enterprise Third-Party Security Assessment Platform**

- Built a full-stack security assessment platform with a deterministic 29-rule risk engine, weighted scoring model, and control-domain mapping inspired by NIST CSF and ISO 27001 to evaluate SaaS, AI tools, IoT platforms, and 4 additional technology categories
- Implemented end-to-end vendor assessment workflows including structured questionnaires, automated findings generation, remediation tracking with owner assignment, and professional PDF report generation with executive summaries
- Designed transparent, interview-explainable risk scoring (inherent and residual) with audit trail logging, role-based access, and optional AI-assisted summarization behind a feature flag, deployed with Docker and GitHub Actions CI

---

## Interview Talking Points

1. **Why I built it**: I wanted to demonstrate that I understand the real-world workflows of an enterprise InfoSec team — not just theoretical knowledge, but the actual process of evaluating vendors, documenting risk, and communicating findings to stakeholders.

2. **How the risk engine works**: The engine uses 29 deterministic rules that check vendor metadata and questionnaire answers. Each rule produces a finding with severity, maps it to a control domain, and calculates a weighted risk score. I chose deterministic logic over ML because risk decisions need to be explainable and auditable.

3. **Why deterministic over AI for scoring**: AI is great for summarization, but risk scores in a governance context must be reproducible and defensible. If a CISO asks "why is this vendor rated High?", you need to point to specific rule logic, not a black-box model. That's why AI in VendorGuard is strictly opt-in and only for narrative generation.

4. **How governance is reflected**: Assessment templates ensure consistency. Every action is audit-logged. Control domains map to recognized frameworks. Reports follow a formal structure. This mirrors how mature organizations maintain governance over their vendor programs.

5. **How remediation tracking works**: Each finding generates a remediation item with priority, assigned owner, and due date. Status progression (open → in progress → mitigated → closed) mirrors ITSM workflows. Residual risk recalculates as items are remediated.

6. **Emerging technology assessment**: I included category-specific controls for AI (data access scope, prompt retention, model training), IoT (segmentation, device inventory, default credentials), tokenization (key management, HSM), and DLT (PII on-chain, privacy controls). These reflect real concerns security teams face with emerging technologies.

7. **How I'd extend it in production**: Evidence attachment storage, multi-reviewer approval chains, vendor tiering with automated review cadence, integration with GRC platforms, and email notifications for approaching due dates.

8. **Technical choices**: FastAPI for async capability and auto-generated API docs, SQLAlchemy for ORM with explicit migrations, Jinja2 for server-rendered templates (realistic for internal tools), and WeasyPrint for PDF generation. Docker Compose for reproducible deployment.

---

*Built as a portfolio project demonstrating readiness for Information Security, Supply Chain Security, and Third-Party Risk roles.*
