# ProcessPilot AI — Business Process Modernization Demo

## Overview

ProcessPilot AI is a full-stack cloud-native application that modernizes how mid-sized enterprises handle internal operational requests. It replaces ad-hoc email threads and spreadsheet tracking with a structured digital workflow powered by intelligent routing, automated prioritization, and AI-generated business summaries.

Built as a portfolio-grade demonstration of consulting-oriented software delivery, the application showcases the complete lifecycle from business requirement analysis through cloud-ready implementation — the kind of work performed on enterprise modernization engagements at firms like IBM, Accenture, and Deloitte.

## Business Problem

A fictional mid-sized enterprise (500–2,000 employees) manages operational requests — IT support, facilities, HR inquiries, procurement, and general operations — through a fragmented process:

- **Email-based submissions** with no standard format, leading to missing information and repeated follow-ups
- **Spreadsheet tracking** maintained by individual managers, creating data silos with no single source of truth
- **Inconsistent prioritization** based on who shouts loudest rather than business impact
- **No visibility** for employees on request status, and no analytics for leadership on operational bottlenecks
- **Delayed response times** because requests fall through the cracks or are routed to the wrong team

The result is lost productivity, employee frustration, compliance risk from untracked requests, and an inability to identify systemic operational issues.

## Solution

ProcessPilot AI replaces this chaos with a structured, intelligent workflow:

1. **Standardized submission** through a web form that captures all required information upfront
2. **Automatic classification** using keyword-based routing logic that assigns requests to the correct department
3. **Data-driven prioritization** using a scoring engine that weighs urgency, business impact, and category
4. **Real-time dashboards** giving employees visibility into their requests and managers oversight of their queues
5. **AI-powered summaries** that help leadership quickly understand complex requests and identify trends
6. **Analytics** that surface operational patterns, bottlenecks, and improvement opportunities

## Architecture

```mermaid
graph TB
    Client[React Frontend] --> API[FastAPI Backend]
    API --> DB[(PostgreSQL)]
    API --> AI[AI Service Layer]
    AI --> OpenAI[OpenAI API]
    AI --> Mock[Mock Provider]
    API --> Router[Routing Engine]
    API --> Priority[Priority Calculator]
```

| Component | Description |
|---|---|
| **React Frontend** | Single-page application with role-based views for employees and managers |
| **FastAPI Backend** | RESTful API handling authentication, business logic, and data access |
| **PostgreSQL** | Relational database storing users, requests, routing decisions, and analytics |
| **AI Service Layer** | Provider abstraction supporting OpenAI and a deterministic mock fallback |
| **Routing Engine** | Rule-based classifier that maps request content to departments and teams |
| **Priority Calculator** | Scoring engine that computes priority from urgency, impact, and category signals |

## Tech Stack

| Technology | Purpose |
|---|---|
| React 18 | Frontend UI framework |
| TypeScript | Type-safe frontend development |
| Vite | Frontend build tool and dev server |
| Tailwind CSS | Utility-first CSS framework |
| FastAPI | High-performance Python API framework |
| SQLAlchemy | Python ORM for database access |
| PostgreSQL 15 | Relational database |
| Alembic | Database migration management |
| Docker & Docker Compose | Containerization and local orchestration |
| Terraform | Infrastructure as Code for AWS deployment |
| JWT (PyJWT) | Stateless authentication tokens |
| OpenAI API | AI-powered request summaries and insights |

## Features

- **Request Submission Portal** — Structured form with category, urgency, and description fields
- **Smart Routing Engine** — Automatic classification and team assignment based on request content
- **Priority Scoring** — Algorithmic prioritization combining urgency, business impact, and category weight
- **AI-Powered Summaries** — On-demand AI analysis of complex requests for leadership review
- **Role-Based Dashboards** — Employees see their requests; managers see their team's queue
- **Real-Time Status Tracking** — Requests move through submitted → in_progress → resolved → closed
- **Manager Review Queue** — Filtered, sortable view of pending requests with bulk actions
- **Analytics Dashboard** — Charts showing request volume trends, category distribution, and resolution times
- **JWT Authentication** — Secure login with role-based access control (employee/manager)
- **Responsive Design** — Mobile-friendly interface built with Tailwind CSS
- **Docker Deployment** — One-command setup with Docker Compose
- **API Documentation** — Auto-generated Swagger/OpenAPI docs

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+ (or Docker)
- Docker & Docker Compose (for containerized setup)

### Local Development Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd "Process Pilot AI"

# 2. Create environment file
cp .env.example .env
# Edit .env with your settings (defaults work for local dev)

# 3. Set up the backend
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Set up the database
# Ensure PostgreSQL is running, then:
alembic upgrade head
python -m app.seed

# 5. Start the backend
uvicorn app.main:app --reload --port 8000

# 6. In a new terminal, set up and start the frontend
cd frontend
npm install
npm run dev
```

The frontend will be available at **http://localhost:5173** and the backend API at **http://localhost:8000**.

### Docker Setup

```bash
cp .env.example .env
docker-compose up --build
```

Access the application at **http://localhost:3000**. The backend API is available at **http://localhost:8000**.

### Demo Credentials

| Email | Password | Role | Description |
|---|---|---|---|
| admin@acme.com | demo123 | Manager | System administrator with full access |
| jsmith@acme.com | demo123 | Employee | Standard employee user |
| mgarcia@acme.com | demo123 | Employee | Standard employee user |
| dkim@acme.com | demo123 | Manager | Department manager |
| ljones@acme.com | demo123 | Employee | Standard employee user |

## Environment Configuration

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/processpilot` |
| `OPENAI_API_KEY` | OpenAI API key for AI features (optional) | *(empty — uses mock provider)* |
| `AI_PROVIDER` | AI provider selection: `openai`, `mock`, or `auto` | `auto` |
| `JWT_SECRET` | Secret key for signing JWT tokens | `change-me-in-production` |
| `APP_ENV` | Environment name: `development`, `staging`, `production` | `development` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## AI Integration

ProcessPilot AI uses a provider abstraction layer for AI functionality, allowing the application to work with or without an OpenAI API key.

### Provider Architecture

- **`auto` mode (default):** Attempts to use OpenAI if an API key is configured; falls back to the mock provider on failure
- **`openai` mode:** Uses the OpenAI API exclusively; returns errors if the key is missing or invalid
- **`mock` mode:** Uses the deterministic mock provider regardless of API key availability

### OpenAI Provider

When configured, the OpenAI provider sends request details to GPT-4 and receives structured summaries including business impact assessment, recommended actions, and complexity ratings. This demonstrates real-world AI integration patterns.

### Mock Provider

The mock provider generates deterministic, realistic-looking summaries based on request metadata (category, urgency, keywords). It produces consistent outputs for testing and demo purposes, ensuring the application is fully functional without any external API dependency.

### Graceful Fallback

The `auto` provider wraps OpenAI calls in error handling and automatically falls back to mock responses if the API is unreachable, rate-limited, or the key is invalid. This pattern mirrors production AI integrations where deterministic business logic must always work, and AI is an enhancement layer.

## API Documentation

After starting the backend, interactive API documentation is available at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Main Endpoint Groups

| Endpoint Group | Description |
|---|---|
| `POST /api/auth/login` | Authenticate and receive a JWT token |
| `GET /api/auth/me` | Get current user profile |
| `GET /api/requests` | List requests (filtered by role) |
| `POST /api/requests` | Submit a new request |
| `GET /api/requests/{id}` | Get request detail with routing info |
| `PATCH /api/requests/{id}` | Update request status or details |
| `POST /api/requests/{id}/summarize` | Generate AI summary for a request |
| `GET /api/analytics/overview` | Dashboard analytics and KPIs |
| `GET /api/analytics/by-category` | Request counts grouped by category |
| `GET /api/analytics/by-department` | Request counts grouped by department |
| `GET /api/analytics/by-priority` | Request counts bucketed by priority |
| `GET /api/analytics/status-distribution` | Request counts by status |
| `GET /api/analytics/top-pain-points` | Highest priority operational issues |
| `GET /api/health` | Health check endpoint |

## Testing

```bash
cd backend
pytest -v
```

### Test Coverage Areas

- **Authentication:** Login flow, JWT token generation and validation, role-based access
- **Request CRUD:** Creation, retrieval, update, and filtering of requests
- **Routing Engine:** Category classification and team assignment logic
- **Priority Calculator:** Score computation across different urgency/impact combinations
- **AI Providers:** Mock provider output format, provider selection logic, fallback behavior
- **API Integration:** End-to-end API tests with test database fixtures

## Screenshots

> Screenshots are generated from the running application and stored in the `screenshots/` directory.

- **Dashboard:** `screenshots/dashboard.png`
- **Request Form:** `screenshots/new-request.png`
- **Request Detail:** `screenshots/request-detail.png`
- **Analytics:** `screenshots/analytics.png`
- **Manager Queue:** `screenshots/manager-queue.png`

## Tradeoffs & Future Improvements

| Current State | Future Evolution |
|---|---|
| Monorepo with shared Docker Compose | Split into separate repos with independent CI/CD pipelines |
| Synchronous SQLAlchemy ORM | Async SQLAlchemy + asyncpg for higher throughput |
| Simple JWT authentication | Enterprise SSO/LDAP integration (SAML, OAuth2) |
| Single-node deployment | Redis caching layer, Celery for async task processing |
| Mock AI with optional OpenAI | Fine-tuned models, retrieval-augmented generation (RAG) |
| Basic role-based access | Granular RBAC with department-level permissions |
| Manual deployment | Full CI/CD with GitHub Actions, automated testing, blue-green deploys |

### Future Feature Roadmap

- Email and Slack notifications on status changes
- SLA tracking with escalation rules
- Multi-level approval workflows
- Comprehensive audit log with change history
- Advanced RBAC with custom permission sets
- Mobile-responsive redesign with PWA support
- Performance optimization with query caching and pagination

## Interview Talking Points

Use these points to discuss the project in technical and behavioral interviews:

1. **Business-to-Technical Translation:** Identified a real operational pain point (email/spreadsheet chaos) and designed a structured technical solution, demonstrating the consulting skill of translating business requirements into architecture decisions.

2. **Cloud-Native Containerization:** Designed the application from day one for containerized deployment with Docker, Docker Compose for local development, and Terraform for cloud infrastructure — showing fluency with modern deployment patterns.

3. **API-First Design:** Built the backend as a standalone RESTful API with FastAPI, enabling independent frontend development, automated documentation (Swagger/OpenAPI), and future mobile or third-party integrations.

4. **Deterministic Logic + AI Augmentation:** The routing engine and priority calculator use deterministic business rules that always work, while AI summaries provide optional intelligence — demonstrating a pragmatic approach to AI integration where core functionality never depends on external services.

5. **Database Design & ORM Patterns:** Designed a normalized relational schema with SQLAlchemy models, Alembic migrations, and seed data — showing proficiency with data modeling and database lifecycle management.

6. **Microservice-Ready Monolith:** Structured the codebase with clear service boundaries (routing engine, priority calculator, AI provider) that could be extracted into independent microservices, balancing pragmatic MVP delivery with future scalability.

7. **Infrastructure as Code:** Created a complete Terraform scaffold for AWS (VPC, ECS Fargate, RDS, ALB) that demonstrates understanding of cloud architecture patterns, security groups, and IAM roles.

8. **Testing Strategy:** Implemented unit tests for business logic, integration tests for API endpoints, and used dependency injection patterns that make the codebase testable — showing awareness of quality engineering practices.

9. **Security Consciousness:** Implemented JWT authentication, CORS configuration, input validation, environment-based secrets management, and network-level security groups — demonstrating security as a first-class concern.

10. **Agile Delivery Mindset:** Organized work into epics, user stories, and sprints with clear MVP scope, deferred features, and a product backlog — showing the ability to plan and communicate delivery incrementally.

11. **Full-Stack Versatility:** Worked across React/TypeScript frontend, Python/FastAPI backend, PostgreSQL database, Docker orchestration, and Terraform infrastructure — demonstrating breadth across the modern development stack.

12. **Documentation as Communication:** Created comprehensive README, architecture docs, business case, and agile notes — recognizing that code alone doesn't convey intent, and documentation is essential for team collaboration.

## Resume Bullets

- Designed and built ProcessPilot AI, a cloud-native business process modernization platform using React, FastAPI, and PostgreSQL with AI-powered request routing and analytics
- Implemented intelligent routing engine and priority scoring system that automates request classification, reducing manual triage time by an estimated 40-60%
- Architected AI provider abstraction layer with OpenAI integration and deterministic fallback, ensuring 100% application availability regardless of external API status
- Created Infrastructure as Code with Terraform for AWS deployment (ECS Fargate, RDS, ALB) demonstrating cloud architecture and DevOps proficiency
- Delivered full-stack application following agile methodology with Docker containerization, JWT authentication, and comprehensive API documentation

## License

MIT
