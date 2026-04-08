# Stakeholder Test Summary Report — Clinical Imaging QA Lab

## Report Information

| Field              | Value                                       |
|--------------------|---------------------------------------------|
| **Project**        | Clinical Imaging QA Lab                     |
| **Version**        | v1.0.0                                      |
| **Report Date**    | YYYY-MM-DD                                  |
| **Prepared By**    | [QA Engineer Name]                          |
| **Test Period**    | YYYY-MM-DD to YYYY-MM-DD                   |

---

## Executive Summary

[2–3 sentence summary of overall testing results — pass rate, critical issues found, and readiness recommendation.]

---

## Test Scope

| Area                  | Tested | Notes                                   |
|-----------------------|--------|-----------------------------------------|
| Capture Workflow      | Yes    | Full CRUD, device interaction, retries  |
| Defect Tracker        | Yes    | Create, list, view                      |
| Dashboard             | Yes    | Summary stats, recent records           |
| Device Simulator      | Yes    | Online, offline, failure modes          |
| API Validation        | Yes    | All endpoints tested                    |
| Database Integrity    | Yes    | SQL validation queries                  |
| Accessibility         | Yes    | axe-core checks, labels, landmarks     |
| Cross-Browser         | Yes    | Chromium, Firefox                       |
| Performance           | Yes    | Locust load testing                     |
| Responsive Design     | Yes    | Desktop, tablet, mobile viewports       |

---

## Test Metrics

| Metric                  | Value  |
|-------------------------|--------|
| Total Test Cases        | XX     |
| Passed                  | XX     |
| Failed                  | XX     |
| Blocked                 | XX     |
| Pass Rate               | XX%    |
| Critical Defects Found  | XX     |
| Major Defects Found     | XX     |
| Minor Defects Found     | XX     |

---

## Test Results by Category

### API Tests
| Suite              | Total | Passed | Failed |
|--------------------|-------|--------|--------|
| Captures           | XX    | XX     | XX     |
| Defects            | XX    | XX     | XX     |
| Device             | XX    | XX     | XX     |
| Dashboard          | XX    | XX     | XX     |

### UI Tests (Playwright)
| Suite              | Total | Passed | Failed |
|--------------------|-------|--------|--------|
| Dashboard          | XX    | XX     | XX     |
| Capture            | XX    | XX     | XX     |
| History            | XX    | XX     | XX     |
| Defects            | XX    | XX     | XX     |

### Cross-Browser (Selenium)
| Browser  | Total | Passed | Failed |
|----------|-------|--------|--------|
| Chrome   | XX    | XX     | XX     |
| Firefox  | XX    | XX     | XX     |

### Accessibility
| Page       | Critical | Serious | Moderate | Minor |
|------------|----------|---------|----------|-------|
| Dashboard  | XX       | XX      | XX       | XX    |
| Capture    | XX       | XX      | XX       | XX    |
| History    | XX       | XX      | XX       | XX    |
| Defects    | XX       | XX      | XX       | XX    |

### Performance
| Metric                    | Value  | Target |
|---------------------------|--------|--------|
| Avg Response Time (ms)    | XX     | <500   |
| 95th Percentile (ms)      | XX     | <1000  |
| Requests/second           | XX     | >50    |
| Error Rate                | XX%    | <1%    |

---

## Key Findings

### Critical Issues
1. [Issue title — brief description]
2. [Issue title — brief description]

### Risks and Concerns
1. [Risk description and potential impact]

---

## Recommendations

- [ ] [Recommendation 1]
- [ ] [Recommendation 2]
- [ ] [Recommendation 3]

---

## Sign-Off

| Role               | Name    | Date       | Approval    |
|--------------------|---------|------------|-------------|
| QA Lead            | [Name]  | YYYY-MM-DD | ☐ Approved  |
| Dev Lead           | [Name]  | YYYY-MM-DD | ☐ Approved  |
| Product Owner      | [Name]  | YYYY-MM-DD | ☐ Approved  |
