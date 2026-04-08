# Test Plan — Clinical Imaging QA Lab

## 1. Purpose
This test plan defines the scope, approach, resources, and schedule for testing the Clinical Imaging QA Lab application. The application simulates a clinical imaging workflow where operators capture, review, and track medical images through a mock hardware device.

## 2. Scope

### In Scope
- Dashboard functionality and data accuracy
- Image capture workflow (form submission → device interaction → result storage)
- Capture history display and data integrity
- Defect tracking (create, list, view)
- Device simulator behavior (online, offline, failure modes)
- API endpoint validation (request/response, error handling)
- Database integrity and SQL validation
- Frontend accessibility (WCAG 2.1 Level AA targets)
- Cross-browser compatibility (Chromium, Firefox)
- Responsive design (desktop, tablet, mobile viewports)
- Performance under simulated load

### Out of Scope
- HIPAA compliance (simulated data only)
- Real device hardware integration
- User authentication and authorization
- Multi-tenant data isolation
- Production deployment and monitoring

## 3. Test Strategy

### 3.1 Unit/API Testing
- **Tool**: Pytest with httpx
- **Coverage**: All backend API endpoints
- **Focus**: Input validation, response shape, status codes, edge cases

### 3.2 Integration Testing
- **Tool**: Pytest with httpx, SQLAlchemy
- **Coverage**: End-to-end flows across backend, device simulator, and database
- **Focus**: Capture → store → retrieve flow, retry logic, dashboard accuracy

### 3.3 SQL Validation Testing
- **Tool**: Pytest with raw SQL assertions
- **Coverage**: Row insertion, error message storage, retry count, dashboard counts
- **Focus**: Database state matches expected outcomes after workflows

### 3.4 UI Testing
- **Tool**: Playwright (primary), Selenium (smoke)
- **Coverage**: All four pages — dashboard, capture, history, defects
- **Focus**: Form submission, validation, navigation, data rendering

### 3.5 Accessibility Testing
- **Tool**: Playwright with axe-core injection
- **Coverage**: All pages
- **Focus**: Critical/serious WCAG violations, labels, landmarks, focus management

### 3.6 Cross-Browser Testing
- **Tool**: Playwright (Chromium, Firefox), Selenium (Chrome, Firefox)
- **Coverage**: Page load, form submission, table rendering

### 3.7 Performance Testing
- **Tool**: Locust
- **Coverage**: Backend API endpoints
- **Focus**: Response times, throughput, error rate under load

## 4. Environment

| Component        | Technology            | Port  |
|------------------|-----------------------|-------|
| Frontend         | HTML/CSS/JS + Nginx   | 8080  |
| Backend API      | Python 3.12 + FastAPI | 8000  |
| Device Simulator | Python 3.12 + FastAPI | 8001  |
| Database         | PostgreSQL 16         | 5432  |

## 5. Test Execution

### Entry Criteria
- All services start without errors
- Database tables are created
- Device simulator is reachable from backend

### Exit Criteria
- All critical and high-priority test cases pass
- No critical accessibility violations
- API response times < 500ms at baseline load
- SQL validation confirms data integrity

## 6. Risks

| Risk                                  | Impact | Mitigation                          |
|---------------------------------------|--------|-------------------------------------|
| Device simulator flaky in random mode | Medium | Use reset endpoint before tests     |
| Database contamination between tests  | Medium | Use unique identifiers per test run |
| Port conflicts on developer machines  | Low    | Document required ports in README   |

## 7. Deliverables
- Automated test suites (see tests/ directory)
- SQL validation queries (see sql/ directory)
- Regression checklist
- Defect report template
- Test case matrix
- This test plan
