# Clinical Imaging QA Lab

A production-style full-stack web application and QA framework simulating a clinical imaging workflow. Built as a portfolio project demonstrating end-to-end development, comprehensive testing, accessibility engineering, and DevOps practices.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       Docker Compose                           │
│                                                                │
│   ┌────────────┐    ┌────────────┐    ┌──────────────────┐     │
│   │  Frontend   │───▶│  Backend   │───▶│ Device Simulator │     │
│   │  (Nginx)    │    │  (FastAPI) │    │    (FastAPI)     │     │
│   │  :8080      │    │  :8000     │    │    :8001         │     │
│   └────────────┘    └──────┬─────┘    └──────────────────┘     │
│                            │                                    │
│                     ┌──────▼──────┐                             │
│                     │ PostgreSQL  │                             │
│                     │   :5432     │                             │
│                     └─────────────┘                             │
└────────────────────────────────────────────────────────────────┘
```

## Features

- **Dashboard** — Real-time device status, capture/defect summary cards, recent activity tables
- **Image Capture** — Form-based capture workflow with live device interaction and instant results
- **Capture History** — Full table with status badges, retry counts, and one-click retry for failures
- **Defect Tracker** — Log bugs with severity, priority, environment, steps, and expected/actual results
- **Mock Device Simulator** — Configurable imaging hardware with online/offline/failure modes
- **SQL Validation** — Direct database assertions verifying data integrity after workflows
- **Automated Testing** — Playwright, Selenium, Pytest, Locust test suites
- **Accessibility** — WCAG 2.1 targets with axe-core checks, semantic HTML, keyboard navigation
- **Docker** — Full-stack containerization with docker-compose
- **CI/CD** — GitHub Actions pipeline with automated test runs

## Tech Stack

| Layer              | Technology                      |
|--------------------|---------------------------------|
| Frontend           | HTML, CSS, vanilla JavaScript   |
| Backend API        | Python 3.12, FastAPI            |
| ORM                | SQLAlchemy                      |
| Database           | PostgreSQL 16                   |
| Device Simulator   | Python 3.12, FastAPI            |
| UI Testing         | Playwright, Selenium            |
| API Testing        | Pytest, httpx                   |
| Accessibility      | axe-core via Playwright         |
| Performance        | Locust                          |
| Containerization   | Docker, docker-compose          |
| CI                 | GitHub Actions                  |

## Folder Structure

```
Clinical Imaging QA Lab/
├── frontend/                    # HTML/CSS/JS client
│   ├── index.html               # Dashboard
│   ├── capture.html             # Capture workflow
│   ├── history.html             # Capture history
│   ├── defects.html             # Defect tracker
│   ├── css/styles.css           # Design system
│   ├── js/                      # Modular JavaScript
│   ├── nginx.conf               # Nginx reverse proxy config
│   └── Dockerfile
├── backend/                     # FastAPI main application
│   ├── app/
│   │   ├── main.py              # App entry point
│   │   ├── config.py            # Environment config
│   │   ├── database.py          # SQLAlchemy setup
│   │   ├── models.py            # ORM models
│   │   ├── schemas.py           # Pydantic schemas
│   │   ├── routers/             # API route modules
│   │   └── services/            # Business logic layer
│   ├── seed_data.py             # Sample data seeder
│   ├── requirements.txt
│   └── Dockerfile
├── device-simulator/            # Mock imaging hardware
│   ├── app/
│   │   ├── main.py              # Simulator endpoints
│   │   ├── device_state.py      # In-memory state
│   │   └── schemas.py           # Request schemas
│   ├── requirements.txt
│   └── Dockerfile
├── tests/                       # All test suites
│   ├── api/                     # Backend API tests
│   ├── integration/             # Cross-service & SQL validation
│   ├── ui/playwright/           # Playwright browser tests
│   ├── ui/selenium/             # Selenium smoke tests
│   ├── accessibility/           # axe-core checks
│   ├── performance/             # Locust load tests
│   ├── conftest.py              # Shared fixtures
│   └── requirements.txt
├── sql/                         # SQL validation queries
│   └── validation_queries.sql
├── qa-docs/                     # QA documentation
│   ├── test-plan.md
│   ├── regression-checklist.md
│   ├── defect-report-template.md
│   ├── test-case-matrix.md
│   └── stakeholder-test-summary-template.md
├── .github/workflows/ci.yml    # GitHub Actions CI
├── docker-compose.yml           # Full-stack orchestration
├── playwright.config.py         # Playwright settings
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

---

## Local Setup

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ (or Docker)
- Node.js 18+ (only for Playwright browser installs)

### Option A: Docker (Recommended)

```bash
# Clone and navigate
cd "Clinical Imaging QA Lab"

# Copy environment file
cp .env.example .env

# Start the full stack
docker-compose up --build

# Access the application
# Frontend: http://localhost:8080
# Backend API: http://localhost:8000
# Device Simulator: http://localhost:8001
```

### Option B: Local Development (Without Docker)

#### 1. Start PostgreSQL

Make sure PostgreSQL is running on port 5432. Create the database:

```bash
# On Windows (PowerShell)
psql -U postgres -c "CREATE USER ciqalab WITH PASSWORD 'ciqalab_pass';"
psql -U postgres -c "CREATE DATABASE ciqalab OWNER ciqalab;"

# On macOS/Linux
sudo -u postgres psql -c "CREATE USER ciqalab WITH PASSWORD 'ciqalab_pass';"
sudo -u postgres psql -c "CREATE DATABASE ciqalab OWNER ciqalab;"
```

#### 2. Set Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Or set directly:

```powershell
# Windows PowerShell
$env:DATABASE_URL = "postgresql://ciqalab:ciqalab_pass@localhost:5432/ciqalab"
$env:DEVICE_SIMULATOR_URL = "http://localhost:8001"
```

```bash
# macOS/Linux
export DATABASE_URL="postgresql://ciqalab:ciqalab_pass@localhost:5432/ciqalab"
export DEVICE_SIMULATOR_URL="http://localhost:8001"
```

#### 3. Start Device Simulator

```bash
cd device-simulator
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

#### 4. Start Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend serves the frontend files, so you can access everything at `http://localhost:8000`.

#### 5. Seed Sample Data (Optional)

```bash
cd backend
python seed_data.py
```

---

## How to Run Tests

### Install Test Dependencies

```bash
python -m pip install -r tests/requirements.txt
```

### API Tests

```bash
python -m pytest tests/api/ -v
```

### Integration Tests (includes SQL validation)

```bash
python -m pytest tests/integration/ -v
```

### Playwright UI Tests

```bash
# Install browsers first
python -m playwright install chromium firefox

# Run tests
python -m pytest tests/ui/playwright/ -v

# Run against specific browser
python -m pytest tests/ui/playwright/ -v --browser chromium
python -m pytest tests/ui/playwright/ -v --browser firefox
```

### Selenium Smoke Tests

Requires Chrome and/or Firefox installed with matching WebDriver:

```bash
python -m pytest tests/ui/selenium/ -v
```

### Accessibility Tests

```bash
python -m pytest tests/accessibility/ -v
```

### Performance Tests (Locust)

```bash
# Start the web UI
python -m locust -f tests/performance/locustfile.py

# Or run headless
python -m locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000
```

**Key metrics to watch:**
- **Response time** — p50 < 200ms, p95 < 500ms for dashboard/history endpoints
- **Throughput** — Target > 50 req/s for read-heavy endpoints
- **Error rate** — Should stay < 1% under baseline load
- **Capture latency** — Depends on device simulator; may be higher with failure modes

### Run All Tests

```bash
python -m pytest tests/api/ tests/integration/ tests/ui/playwright/ tests/accessibility/ -v
```

---

## API Reference

| Method | Endpoint                        | Description                        |
|--------|---------------------------------|------------------------------------|
| GET    | `/api/health`                   | Health check                       |
| GET    | `/api/device/status`            | Device status (proxied)            |
| GET    | `/api/dashboard/summary`        | Dashboard aggregate data           |
| POST   | `/api/captures`                 | Create new capture                 |
| GET    | `/api/captures`                 | List all captures                  |
| GET    | `/api/captures/{id}`            | Get capture by ID                  |
| POST   | `/api/captures/{id}/retry`      | Retry failed capture               |
| POST   | `/api/defects`                  | Create new defect                  |
| GET    | `/api/defects`                  | List all defects                   |
| GET    | `/api/defects/{id}`             | Get defect by ID                   |

### Sample API Flows

**Create a capture:**
```bash
curl -X POST http://localhost:8000/api/captures \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "PAT-001", "session_id": "SESS-001", "image_type": "x-ray"}'
```

**Log a defect:**
```bash
curl -X POST http://localhost:8000/api/defects \
  -H "Content-Type: application/json" \
  -d '{"title": "Button misaligned", "severity": "minor", "priority": "low"}'
```

**Check dashboard:**
```bash
curl http://localhost:8000/api/dashboard/summary
```

---

## Inspecting the Database

Connect to PostgreSQL directly:

```bash
# Docker
docker exec -it clinical-imaging-qa-lab-postgres-1 psql -U ciqalab -d ciqalab

# Local
psql -U ciqalab -d ciqalab
```

Run validation queries from `sql/validation_queries.sql`:

```sql
SELECT capture_status, COUNT(*) FROM captures GROUP BY capture_status;
SELECT severity, COUNT(*) FROM defects GROUP BY severity;
```

---

## QA Strategy

This project emphasizes quality assurance at every layer:

1. **API Tests** — Validate all endpoints, input validation, error handling
2. **Integration Tests** — End-to-end flows across backend ↔ device ↔ database
3. **SQL Validation** — Direct database assertions after key workflows
4. **UI Tests** — Playwright tests for user journeys across all pages
5. **Smoke Tests** — Selenium tests for cross-browser baseline
6. **Accessibility** — axe-core scans, label association, focus management
7. **Performance** — Locust load testing with configurable scenarios

See `qa-docs/` for detailed documentation:
- [Test Plan](qa-docs/test-plan.md)
- [Regression Checklist](qa-docs/regression-checklist.md)
- [Test Case Matrix](qa-docs/test-case-matrix.md)
- [Defect Report Template](qa-docs/defect-report-template.md)
- [Stakeholder Summary Template](qa-docs/stakeholder-test-summary-template.md)

## Accessibility Strategy

The frontend implements the following accessibility practices:

- **Skip navigation** — Skip-to-content link on every page
- **Semantic HTML** — Proper heading hierarchy, landmarks, form associations
- **ARIA attributes** — `aria-live` for dynamic content, `aria-required` for forms, `role` attributes
- **Keyboard navigation** — All interactive elements are keyboard-accessible
- **Visible focus** — Custom `:focus-visible` styles with contrasting outlines
- **Color contrast** — Design tokens chosen for WCAG 2.1 AA compliance
- **Error messaging** — Inline validation errors with `role="alert"` for screen readers
- **Responsive** — Tables scroll horizontally on small viewports; mobile nav toggle

---

## Known Assumptions

- This is a **simulation** — no real DICOM images or medical hardware involved
- Patient data is fictional; no PHI/HIPAA considerations apply
- The device simulator runs in-memory; state resets on restart
- The `random_failure` mode is non-deterministic — tests use `reset` before assertions
- Authentication/authorization is out of scope for this version
- PostgreSQL must be accessible on the configured port before backend starts

---

## License

This project is a portfolio demonstration. No license restrictions.
