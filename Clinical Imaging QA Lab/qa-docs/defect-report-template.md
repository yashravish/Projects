# Defect Report Template — Clinical Imaging QA Lab

## Defect Summary

| Field          | Value                                       |
|----------------|---------------------------------------------|
| **Defect ID**  | DEF-XXXX                                    |
| **Title**      | [Brief, descriptive summary]                |
| **Severity**   | Critical / Major / Minor / Trivial          |
| **Priority**   | High / Medium / Low                         |
| **Status**     | Open / In Progress / Resolved / Closed      |
| **Reporter**   | [Name]                                      |
| **Date Found** | YYYY-MM-DD                                  |
| **Environment**| [Browser, OS, service versions]             |

## Description
[Detailed description of the defect, including context of what was being tested.]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Observe the defect]

## Expected Result
[What should have happened]

## Actual Result
[What actually happened]

## Evidence
- Screenshot(s): [Attach or link]
- Console errors: [Paste relevant console output]
- API response: [Paste relevant response body if applicable]
- Log excerpt: [Paste relevant server log lines]

## Impact
[Describe the impact on the user or system. Does it block other testing?]

## Workaround
[If a workaround exists, describe it. Otherwise write "None known."]

## Resolution Notes
[To be filled when the defect is resolved — what was changed and why.]

---

### Example

| Field          | Value                                                |
|----------------|------------------------------------------------------|
| **Defect ID**  | DEF-0012                                             |
| **Title**      | Capture timeout error shows generic message to user  |
| **Severity**   | Major                                                |
| **Priority**   | High                                                 |
| **Status**     | Open                                                 |
| **Reporter**   | QA Engineer                                          |
| **Date Found** | 2026-04-08                                           |
| **Environment**| Chrome 120, Windows 11, Backend v1.0.0               |

**Steps to Reproduce:**
1. Set device simulator to timeout failure mode
2. Navigate to Capture page
3. Fill in patient and session details
4. Click "Start Capture"
5. Observe the error message

**Expected:** Error message says "Capture timed out — device did not respond within 10 seconds"
**Actual:** Error message says "Unknown device error"
