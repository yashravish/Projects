# FlightGuard Traceability Matrix

**Document ID:** FG-TM-001  
**Version:** 1.0  
**Project:** FlightGuard Avionics Sensor Health Monitor  

---

## Purpose

This matrix maps each software requirement to its implementing function(s)
and verifying test case(s). Full bidirectional traceability ensures that:
- Every requirement has implementation coverage
- Every requirement has test coverage
- Every test traces back to a requirement

---

## Requirement → Implementation → Test Case(s)

| Requirement | Description | Implementation Function(s) | Test Case(s) |
|-------------|-------------|---------------------------|--------------|
| FGR-REQ-001 | Airspeed out-of-range sets INVALID_RANGE | `fg_is_airspeed_in_range()`, `fg_are_ranges_valid()` | TC-UNIT-004, TC-UNIT-005, TC-UNIT-034, TC-UNIT-035, TC-UNIT-028, TC-FAULT-002, TC-FAULT-003, TC-FAULT-006, TC-FAULT-007 |
| FGR-REQ-002 | Data age > 2000ms sets DATA_STALE | `fg_is_data_stale()` | TC-UNIT-010, TC-UNIT-011, TC-UNIT-012, TC-UNIT-013, TC-FAULT-004, TC-FAULT-012, TC-FAULT-020, TC-INTEG-002 |
| FGR-REQ-002a | Clock anomaly treated as stale | `fg_is_data_stale()` | TC-UNIT-014 |
| FGR-REQ-003 | sensor_valid=false triggers FAILSAFE | `fg_evaluate()`, `fg_determine_mode()` | TC-UNIT-023, TC-FAULT-001, TC-INTEG-004 |
| FGR-REQ-004 | Airspeed > 340kt sets OVERSPEED | `fg_is_overspeed()` | TC-UNIT-019, TC-UNIT-020, TC-FAULT-015, TC-INTEG-006 |
| FGR-REQ-005 | AoA > 15deg sets STALL_RISK | `fg_is_stall_risk()` | TC-UNIT-021, TC-UNIT-022, TC-FAULT-016 |
| FGR-REQ-006 | \|primary-redundant\| > 15kt sets DISAGREEMENT | `fg_is_disagreement_detected()` | TC-UNIT-015, TC-UNIT-016, TC-UNIT-017, TC-UNIT-018, TC-UNIT-039, TC-FAULT-005, TC-FAULT-013, TC-FAULT-014, TC-INTEG-003 |
| FGR-REQ-007 | 1 health warning → DEGRADED | `fg_determine_mode()` | TC-UNIT-024, TC-INTEG-002, TC-INTEG-003, TC-INTEG-004, TC-INTEG-005, TC-FAULT-002 |
| FGR-REQ-008 | 2+ health warnings → FAILSAFE | `fg_determine_mode()` | TC-UNIT-025, TC-UNIT-038, TC-INTEG-005, TC-FAULT-017, TC-FAULT-018, TC-FAULT-019 |
| FGR-REQ-009 | No health warnings → NORMAL | `fg_determine_mode()` | TC-UNIT-001, TC-INTEG-001, TC-INTEG-002 |
| FGR-REQ-010 | Altitude out-of-range sets INVALID_RANGE | `fg_is_altitude_in_range()`, `fg_are_ranges_valid()` | TC-UNIT-006, TC-UNIT-007, TC-UNIT-036, TC-UNIT-037, TC-FAULT-008, TC-FAULT-009, TC-INTEG-004 |
| FGR-REQ-011 | AoA out-of-range sets INVALID_RANGE | `fg_is_aoa_in_range()`, `fg_are_ranges_valid()` | TC-UNIT-008, TC-UNIT-009, TC-FAULT-010, TC-FAULT-011 |
| FGR-REQ-012 | No dynamic memory allocation | All functions (design constraint) | Verified by code review |
| FGR-REQ-013 | Deterministic output | `fg_evaluate()` | TC-UNIT-040 |
| FGR-REQ-014 | Flight warnings don't affect mode | `fg_determine_mode()` | TC-UNIT-026, TC-UNIT-027, TC-INTEG-006 |
| FGR-REQ-015 | NULL pointers return error | `fg_evaluate()` | TC-UNIT-002, TC-UNIT-003 |

---

## Reverse Trace: Test → Requirement(s)

| Test Case | Requirement(s) Verified |
|-----------|------------------------|
| TC-UNIT-001 | FGR-REQ-009 |
| TC-UNIT-002 | FGR-REQ-015 |
| TC-UNIT-003 | FGR-REQ-015 |
| TC-UNIT-004 | FGR-REQ-001 |
| TC-UNIT-005 | FGR-REQ-001 |
| TC-UNIT-006 | FGR-REQ-010 |
| TC-UNIT-007 | FGR-REQ-010 |
| TC-UNIT-008 | FGR-REQ-011 |
| TC-UNIT-009 | FGR-REQ-011 |
| TC-UNIT-010 | FGR-REQ-002 |
| TC-UNIT-011 | FGR-REQ-002 |
| TC-UNIT-012 | FGR-REQ-002 |
| TC-UNIT-013 | FGR-REQ-002 |
| TC-UNIT-014 | FGR-REQ-002a |
| TC-UNIT-015 | FGR-REQ-006 |
| TC-UNIT-016 | FGR-REQ-006 |
| TC-UNIT-017 | FGR-REQ-006 |
| TC-UNIT-018 | FGR-REQ-006 |
| TC-UNIT-019 | FGR-REQ-004 |
| TC-UNIT-020 | FGR-REQ-004 |
| TC-UNIT-021 | FGR-REQ-005 |
| TC-UNIT-022 | FGR-REQ-005 |
| TC-UNIT-023 | FGR-REQ-003 |
| TC-UNIT-024 | FGR-REQ-007 |
| TC-UNIT-025 | FGR-REQ-008 |
| TC-UNIT-026 | FGR-REQ-014 |
| TC-UNIT-027 | FGR-REQ-014 |
| TC-UNIT-028 | FGR-REQ-001, FGR-REQ-004, FGR-REQ-005 |
| TC-UNIT-029 | Utility (mode string) |
| TC-UNIT-030 | Utility (warning string) |
| TC-UNIT-031 | Utility (warning string) |
| TC-UNIT-032 | Defensive (NULL buffer) |
| TC-UNIT-033 | Defensive (zero buffer) |
| TC-UNIT-034 | FGR-REQ-001 |
| TC-UNIT-035 | FGR-REQ-001 |
| TC-UNIT-036 | FGR-REQ-010 |
| TC-UNIT-037 | FGR-REQ-010 |
| TC-UNIT-038 | FGR-REQ-008 |
| TC-UNIT-039 | FGR-REQ-006 |
| TC-UNIT-040 | FGR-REQ-013 |
| TC-INTEG-001 | FGR-REQ-009 |
| TC-INTEG-002 | FGR-REQ-002, FGR-REQ-007, FGR-REQ-009 |
| TC-INTEG-003 | FGR-REQ-006, FGR-REQ-007 |
| TC-INTEG-004 | FGR-REQ-010, FGR-REQ-007, FGR-REQ-003 |
| TC-INTEG-005 | FGR-REQ-002, FGR-REQ-006, FGR-REQ-007, FGR-REQ-008 |
| TC-INTEG-006 | FGR-REQ-004, FGR-REQ-014 |
| TC-FAULT-001 | FGR-REQ-003 |
| TC-FAULT-002 | FGR-REQ-001 |
| TC-FAULT-003 | FGR-REQ-001 |
| TC-FAULT-004 | FGR-REQ-002 |
| TC-FAULT-005 | FGR-REQ-006 |
| TC-FAULT-006 | FGR-REQ-001 |
| TC-FAULT-007 | FGR-REQ-001 |
| TC-FAULT-008 | FGR-REQ-010 |
| TC-FAULT-009 | FGR-REQ-010 |
| TC-FAULT-010 | FGR-REQ-011 |
| TC-FAULT-011 | FGR-REQ-011 |
| TC-FAULT-012 | FGR-REQ-002 |
| TC-FAULT-013 | FGR-REQ-006 |
| TC-FAULT-014 | FGR-REQ-006 |
| TC-FAULT-015 | FGR-REQ-004 |
| TC-FAULT-016 | FGR-REQ-005 |
| TC-FAULT-017 | FGR-REQ-002, FGR-REQ-006, FGR-REQ-008 |
| TC-FAULT-018 | FGR-REQ-002, FGR-REQ-001, FGR-REQ-008 |
| TC-FAULT-019 | FGR-REQ-001, FGR-REQ-006, FGR-REQ-008 |
| TC-FAULT-020 | FGR-REQ-002 |

---

## Coverage Summary

| Requirement | Unit Tests | Integration Tests | Fault Tests | Total |
|-------------|-----------|-------------------|-------------|-------|
| FGR-REQ-001 | 4 | 0 | 4 | 8 |
| FGR-REQ-002 | 4 | 1 | 3 | 8 |
| FGR-REQ-002a | 1 | 0 | 0 | 1 |
| FGR-REQ-003 | 1 | 1 | 1 | 3 |
| FGR-REQ-004 | 2 | 1 | 1 | 4 |
| FGR-REQ-005 | 2 | 0 | 1 | 3 |
| FGR-REQ-006 | 5 | 1 | 3 | 9 |
| FGR-REQ-007 | 1 | 4 | 1 | 6 |
| FGR-REQ-008 | 2 | 1 | 3 | 6 |
| FGR-REQ-009 | 1 | 2 | 0 | 3 |
| FGR-REQ-010 | 4 | 1 | 2 | 7 |
| FGR-REQ-011 | 2 | 0 | 2 | 4 |
| FGR-REQ-012 | — | — | — | Code review |
| FGR-REQ-013 | 1 | 0 | 0 | 1 |
| FGR-REQ-014 | 2 | 1 | 0 | 3 |
| FGR-REQ-015 | 2 | 0 | 0 | 2 |
| **Total** | **34** | **12** | **21** | **67+** |

All functional requirements have at least one test case.  
Design constraint FGR-REQ-012 is verified by code review (no calls to malloc/calloc/realloc/free).
