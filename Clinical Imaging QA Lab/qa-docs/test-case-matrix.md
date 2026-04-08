# Test Case Matrix — Clinical Imaging QA Lab

## Capture Workflow Test Cases

| ID     | Test Case                                  | Type        | Priority | Expected Outcome                                | Automated |
|--------|--------------------------------------------|-------------|----------|--------------------------------------------------|-----------|
| TC-001 | Submit capture with valid patient data     | Functional  | High     | Capture created with success/failed status       | Yes       |
| TC-002 | Submit capture with empty patient ID       | Validation  | High     | 422 error / frontend validation message          | Yes       |
| TC-003 | Submit capture with empty session ID       | Validation  | High     | 422 error / frontend validation message          | Yes       |
| TC-004 | Submit capture with no image type          | Validation  | High     | 422 error / frontend validation message          | Yes       |
| TC-005 | Submit capture with invalid image type     | Validation  | Medium   | 422 error returned                               | Yes       |
| TC-006 | Submit capture when device is offline      | Integration | High     | Capture status = failed, error message stored    | Yes       |
| TC-007 | Retry a failed capture                     | Functional  | High     | retry_count incremented, new attempt made         | Yes       |
| TC-008 | Retry a successful capture (no-op)         | Edge Case   | Medium   | Capture returned unchanged                       | Yes       |
| TC-009 | Retry nonexistent capture                  | Error       | Medium   | 404 returned                                     | Yes       |
| TC-010 | Capture with timeout failure mode          | Fault       | Medium   | Capture fails with timeout error                 | Yes       |
| TC-011 | Capture with random failure mode           | Fault       | Medium   | Capture succeeds or fails randomly               | Partial   |
| TC-012 | Capture with unavailable failure mode      | Fault       | Medium   | 503 error from device                            | Yes       |

## Defect Tracker Test Cases

| ID     | Test Case                                  | Type        | Priority | Expected Outcome                                | Automated |
|--------|--------------------------------------------|-------------|----------|--------------------------------------------------|-----------|
| TC-020 | Submit defect with all fields              | Functional  | High     | Defect created with status "open"                | Yes       |
| TC-021 | Submit defect with only required fields    | Functional  | Medium   | Defect created, optional fields null             | Yes       |
| TC-022 | Submit defect with missing title           | Validation  | High     | 422 error / frontend validation message          | Yes       |
| TC-023 | Submit defect with invalid severity        | Validation  | Medium   | 422 error                                        | Yes       |
| TC-024 | Submit defect with invalid priority        | Validation  | Medium   | 422 error                                        | Yes       |
| TC-025 | List defects returns array                 | Functional  | High     | 200 with JSON array                              | Yes       |
| TC-026 | Get single defect by ID                    | Functional  | Medium   | 200 with matching defect                         | Yes       |
| TC-027 | Get nonexistent defect                     | Error       | Medium   | 404 returned                                     | Yes       |

## Dashboard Test Cases

| ID     | Test Case                                  | Type        | Priority | Expected Outcome                                | Automated |
|--------|--------------------------------------------|-------------|----------|--------------------------------------------------|-----------|
| TC-030 | Dashboard loads summary data               | Functional  | High     | All count fields present and non-negative        | Yes       |
| TC-031 | Dashboard counts match database            | Integration | High     | API counts == SQL counts                         | Yes       |
| TC-032 | Recent captures limited to 5               | Functional  | Medium   | Array length <= 5                                | Yes       |
| TC-033 | Recent defects limited to 5                | Functional  | Medium   | Array length <= 5                                | Yes       |
| TC-034 | Device status shown on dashboard           | UI          | High     | Status badge visible with correct state          | Yes       |

## Accessibility Test Cases

| ID     | Test Case                                  | Type          | Priority | Expected Outcome                              | Automated |
|--------|--------------------------------------------|---------------|----------|------------------------------------------------|-----------|
| TC-040 | Skip link present on all pages             | Accessibility | High     | .skip-link element with href to main content  | Yes       |
| TC-041 | Form labels associated with inputs         | Accessibility | High     | All inputs have matching label[for]            | Yes       |
| TC-042 | Visible focus indicators                   | Accessibility | High     | Focus ring/shadow on focused elements          | Yes       |
| TC-043 | No critical axe-core violations            | Accessibility | High     | 0 critical/serious violations per page         | Yes       |
| TC-044 | Semantic landmarks present                 | Accessibility | Medium   | header, main, footer, nav with aria-label      | Yes       |

## Cross-Browser Test Cases

| ID     | Test Case                                  | Type         | Priority | Expected Outcome                               | Automated |
|--------|--------------------------------------------|--------------|----------|--------------------------------------------------|-----------|
| TC-050 | Dashboard loads in Chrome                  | Compatibility| High     | Page loads, data renders                        | Yes       |
| TC-051 | Dashboard loads in Firefox                 | Compatibility| High     | Page loads, data renders                        | Yes       |
| TC-052 | Capture form works in Chrome               | Compatibility| High     | Form submits and shows result                   | Yes       |
| TC-053 | All pages load in Firefox                  | Compatibility| Medium   | All pages render correctly                      | Yes       |

## SQL Validation Test Cases

| ID     | Test Case                                  | Type         | Priority | Expected Outcome                               | Automated |
|--------|--------------------------------------------|--------------|----------|--------------------------------------------------|-----------|
| TC-060 | Capture insert creates row                 | Data         | High     | Row count increases by 1                        | Yes       |
| TC-061 | Failed capture stores error_message        | Data         | High     | error_message IS NOT NULL for failed captures   | Yes       |
| TC-062 | Retry increments retry_count               | Data         | High     | retry_count >= 1 after retry                    | Yes       |
| TC-063 | Defect insert creates row                  | Data         | High     | Row count increases by 1                        | Yes       |
| TC-064 | Dashboard counts match SQL                 | Data         | High     | API summary counts == raw SQL counts            | Yes       |
| TC-065 | Device events logged on capture            | Data         | Medium   | device_events row count increases               | Yes       |
