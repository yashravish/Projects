# FlightGuard Test Cases

**Document ID:** FG-TC-001  
**Version:** 1.0  
**Project:** FlightGuard Avionics Sensor Health Monitor  

---

## 1. Unit Test Cases

### TC-UNIT-001: Nominal Evaluation
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-001 |
| **Requirement** | FGR-REQ-009 |
| **Objective** | Verify NORMAL mode with all valid inputs |
| **Inputs** | airspeed=250.0, alt=35000.0, aoa=3.5, valid=true, ts=1000, cur=1500, red=true, red_as=252.0 |
| **Expected** | Mode=NORMAL, Warnings=NONE, rc=FG_OK |

### TC-UNIT-002: NULL Input Pointer
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-002 |
| **Requirement** | FGR-REQ-015 |
| **Objective** | Verify error return for NULL input |
| **Inputs** | input=NULL |
| **Expected** | rc=FG_ERR_NULL |

### TC-UNIT-003: NULL Result Pointer
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-003 |
| **Requirement** | FGR-REQ-015 |
| **Objective** | Verify error return for NULL result |
| **Inputs** | result=NULL |
| **Expected** | rc=FG_ERR_NULL |

### TC-UNIT-004: Airspeed Below Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-004 |
| **Requirement** | FGR-REQ-001 |
| **Objective** | Verify INVALID_RANGE for airspeed < 0 |
| **Inputs** | airspeed=-1.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-005: Airspeed Above Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-005 |
| **Requirement** | FGR-REQ-001 |
| **Objective** | Verify INVALID_RANGE for airspeed > 500 |
| **Inputs** | airspeed=501.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-006: Altitude Below Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-006 |
| **Requirement** | FGR-REQ-010 |
| **Objective** | Verify INVALID_RANGE for altitude < -1000 |
| **Inputs** | altitude=-1001.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-007: Altitude Above Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-007 |
| **Requirement** | FGR-REQ-010 |
| **Objective** | Verify INVALID_RANGE for altitude > 60000 |
| **Inputs** | altitude=60001.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-008: AoA Below Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-008 |
| **Requirement** | FGR-REQ-011 |
| **Objective** | Verify INVALID_RANGE for AoA < -10 |
| **Inputs** | aoa=-11.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-009: AoA Above Range
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-009 |
| **Requirement** | FGR-REQ-011 |
| **Objective** | Verify INVALID_RANGE for AoA > 40 |
| **Inputs** | aoa=41.0 (others nominal) |
| **Expected** | INVALID_RANGE flag set |

### TC-UNIT-010: Fresh Data
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-010 |
| **Requirement** | FGR-REQ-002 |
| **Objective** | Verify DATA_STALE is NOT set for fresh data |
| **Inputs** | ts=1000, cur=2000 (age=1000ms) |
| **Expected** | DATA_STALE flag NOT set |

### TC-UNIT-011: Stale Data
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-011 |
| **Requirement** | FGR-REQ-002 |
| **Objective** | Verify DATA_STALE for data exceeding threshold |
| **Inputs** | ts=1000, cur=4000 (age=3000ms) |
| **Expected** | DATA_STALE flag set |

### TC-UNIT-012: Stale Boundary (Not Stale)
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-012 |
| **Requirement** | FGR-REQ-002 |
| **Objective** | Verify age exactly at threshold is NOT stale |
| **Inputs** | ts=1000, cur=3000 (age=2000ms) |
| **Expected** | DATA_STALE flag NOT set |

### TC-UNIT-013: Stale Boundary (Stale)
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-013 |
| **Requirement** | FGR-REQ-002 |
| **Objective** | Verify age one ms over threshold IS stale |
| **Inputs** | ts=1000, cur=3001 (age=2001ms) |
| **Expected** | DATA_STALE flag set |

### TC-UNIT-014: Clock Anomaly
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-014 |
| **Requirement** | FGR-REQ-002a |
| **Objective** | Verify clock rollback treated as stale |
| **Inputs** | ts=5000, cur=3000 |
| **Expected** | DATA_STALE flag set |

### TC-UNIT-015: Disagreement Detected
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-015 |
| **Requirement** | FGR-REQ-006 |
| **Objective** | Verify disagreement when diff > tolerance |
| **Inputs** | airspeed=250.0, redundant=270.0 (diff=20) |
| **Expected** | SENSOR_DISAGREEMENT flag set |

### TC-UNIT-016: Disagreement Within Tolerance
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-016 |
| **Requirement** | FGR-REQ-006 |
| **Objective** | Verify no disagreement when diff <= tolerance |
| **Inputs** | airspeed=250.0, redundant=260.0 (diff=10) |
| **Expected** | SENSOR_DISAGREEMENT flag NOT set |

### TC-UNIT-017: Disagreement Boundary Exact
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-017 |
| **Requirement** | FGR-REQ-006 |
| **Objective** | Verify diff exactly at tolerance is NOT triggered |
| **Inputs** | airspeed=250.0, redundant=265.0 (diff=15.0) |
| **Expected** | SENSOR_DISAGREEMENT flag NOT set |

### TC-UNIT-018: No Redundant Available
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-018 |
| **Requirement** | FGR-REQ-006 |
| **Objective** | Verify no disagreement check without redundant |
| **Inputs** | redundant_available=false |
| **Expected** | SENSOR_DISAGREEMENT flag NOT set |

### TC-UNIT-019: Overspeed Detected
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-019 |
| **Requirement** | FGR-REQ-004 |
| **Objective** | Verify overspeed flag above threshold |
| **Inputs** | airspeed=350.0 |
| **Expected** | OVERSPEED flag set |

### TC-UNIT-020: Overspeed Boundary
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-020 |
| **Requirement** | FGR-REQ-004 |
| **Objective** | Verify overspeed NOT set at exact threshold |
| **Inputs** | airspeed=340.0 |
| **Expected** | OVERSPEED flag NOT set |

### TC-UNIT-021: Stall Risk Detected
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-021 |
| **Requirement** | FGR-REQ-005 |
| **Objective** | Verify stall risk flag above threshold |
| **Inputs** | aoa=16.0 |
| **Expected** | STALL_RISK flag set |

### TC-UNIT-022: Stall Risk Boundary
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-022 |
| **Requirement** | FGR-REQ-005 |
| **Objective** | Verify stall risk NOT set at exact threshold |
| **Inputs** | aoa=15.0 |
| **Expected** | STALL_RISK flag NOT set |

### TC-UNIT-023: Sensor Invalid -> FAILSAFE
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-023 |
| **Requirement** | FGR-REQ-003 |
| **Objective** | Verify FAILSAFE on sensor_valid=false |
| **Inputs** | sensor_valid=false |
| **Expected** | Mode=FAILSAFE, Warnings=NONE |

### TC-UNIT-024: One Health Warning -> DEGRADED
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-024 |
| **Requirement** | FGR-REQ-007 |
| **Objective** | Verify DEGRADED with one health warning |
| **Inputs** | ts=1000, cur=4000 (stale) |
| **Expected** | Mode=DEGRADED |

### TC-UNIT-025: Two Health Warnings -> FAILSAFE
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-025 |
| **Requirement** | FGR-REQ-008 |
| **Objective** | Verify FAILSAFE with two health warnings |
| **Inputs** | stale data + disagreement |
| **Expected** | Mode=FAILSAFE, DATA_STALE+DISAGREEMENT set |

### TC-UNIT-026: Overspeed Alone -> NORMAL Mode
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-026 |
| **Requirement** | FGR-REQ-014 |
| **Objective** | Verify overspeed doesn't change mode |
| **Inputs** | airspeed=350.0, all else nominal |
| **Expected** | Mode=NORMAL, OVERSPEED flag set |

### TC-UNIT-027: Stall Risk Alone -> NORMAL Mode
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-027 |
| **Requirement** | FGR-REQ-014 |
| **Objective** | Verify stall risk doesn't change mode |
| **Inputs** | aoa=18.0, all else nominal |
| **Expected** | Mode=NORMAL, STALL_RISK flag set |

### TC-UNIT-028: Invalid Range Skips Flight Checks
| Field | Value |
|-------|-------|
| **Test ID** | TC-UNIT-028 |
| **Requirement** | FGR-REQ-001, FGR-REQ-004, FGR-REQ-005 |
| **Objective** | Verify overspeed/stall not checked with invalid range |
| **Inputs** | airspeed=600.0, aoa=25.0 |
| **Expected** | INVALID_RANGE set, OVERSPEED and STALL_RISK NOT set |

### TC-UNIT-029 through TC-UNIT-040
Additional tests cover: mode string conversion, warning string formatting,
NULL buffer handling, zero buffer size, boundary min/max values for airspeed
and altitude, three simultaneous health warnings, negative disagreement
direction, and determinism verification.

---

## 2. Integration Test Cases

### TC-INTEG-001: Nominal Climb
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-001 |
| **Requirement** | FGR-REQ-009 |
| **Objective** | Verify NORMAL mode through complete nominal flight |
| **Inputs** | 5-step climb from takeoff to cruise |
| **Expected** | NORMAL at every step, no warnings |

### TC-INTEG-002: Cruise Stale Sensor
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-002 |
| **Requirement** | FGR-REQ-002, FGR-REQ-007, FGR-REQ-009 |
| **Objective** | Verify NORMAL -> DEGRADED -> NORMAL on stale recovery |
| **Inputs** | 3-step: normal cruise, sensor freezes, sensor recovers |
| **Expected** | NORMAL -> DEGRADED -> NORMAL |

### TC-INTEG-003: Redundant Disagreement
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-003 |
| **Requirement** | FGR-REQ-006, FGR-REQ-007 |
| **Objective** | Verify disagreement detection as redundant drifts |
| **Inputs** | 3-step: agree, near tolerance, exceed tolerance |
| **Expected** | NORMAL -> NORMAL -> DEGRADED |

### TC-INTEG-004: Invalid Sensor on Approach
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-004 |
| **Requirement** | FGR-REQ-010, FGR-REQ-007, FGR-REQ-003 |
| **Objective** | Verify NORMAL -> DEGRADED -> FAILSAFE escalation |
| **Inputs** | 3-step: normal, altitude glitch, sensor invalid |
| **Expected** | NORMAL -> DEGRADED -> FAILSAFE |

### TC-INTEG-005: Complete Sensor Failure
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-005 |
| **Requirement** | FGR-REQ-002, FGR-REQ-006, FGR-REQ-007, FGR-REQ-008 |
| **Objective** | Verify dual-fault FAILSAFE escalation |
| **Inputs** | 3-step: normal, stale only, stale + disagreement |
| **Expected** | NORMAL -> DEGRADED -> FAILSAFE |

### TC-INTEG-006: Overspeed During Descent
| Field | Value |
|-------|-------|
| **Test ID** | TC-INTEG-006 |
| **Requirement** | FGR-REQ-004, FGR-REQ-014 |
| **Objective** | Verify mode stays NORMAL during overspeed event |
| **Inputs** | 3-step: normal descent, overspeed, speed reduces |
| **Expected** | NORMAL throughout, OVERSPEED flag set only in step 2 |

---

## 3. Fault Injection Test Cases

### TC-FAULT-001 through TC-FAULT-020

| Test ID | Requirement(s) | Injected Fault | Expected Result |
|---------|----------------|----------------|-----------------|
| TC-FAULT-001 | FGR-REQ-003 | sensor_valid=false | FAILSAFE |
| TC-FAULT-002 | FGR-REQ-001 | airspeed=-50.0 | INVALID_RANGE, DEGRADED |
| TC-FAULT-003 | FGR-REQ-001 | airspeed=99999.0 | INVALID_RANGE |
| TC-FAULT-004 | FGR-REQ-002 | age=1000000ms | DATA_STALE |
| TC-FAULT-005 | FGR-REQ-006 | diff=150kt | SENSOR_DISAGREEMENT |
| TC-FAULT-006 | FGR-REQ-001 | airspeed=-0.1 | INVALID_RANGE |
| TC-FAULT-007 | FGR-REQ-001 | airspeed=500.1 | INVALID_RANGE |
| TC-FAULT-008 | FGR-REQ-010 | altitude=-1000.1 | INVALID_RANGE |
| TC-FAULT-009 | FGR-REQ-010 | altitude=60000.1 | INVALID_RANGE |
| TC-FAULT-010 | FGR-REQ-011 | aoa=-10.1 | INVALID_RANGE |
| TC-FAULT-011 | FGR-REQ-011 | aoa=40.1 | INVALID_RANGE |
| TC-FAULT-012 | FGR-REQ-002 | age=2000ms exactly | NOT stale |
| TC-FAULT-013 | FGR-REQ-006 | diff=15.0kt exactly | NOT triggered |
| TC-FAULT-014 | FGR-REQ-006 | diff=15.1kt | SENSOR_DISAGREEMENT |
| TC-FAULT-015 | FGR-REQ-004 | airspeed=340.1 | OVERSPEED |
| TC-FAULT-016 | FGR-REQ-005 | aoa=15.1 | STALL_RISK |
| TC-FAULT-017 | FGR-REQ-002, 006, 008 | stale + disagreement | FAILSAFE |
| TC-FAULT-018 | FGR-REQ-002, 001, 008 | stale + invalid range | FAILSAFE |
| TC-FAULT-019 | FGR-REQ-001, 006, 008 | invalid range + disagreement | FAILSAFE |
| TC-FAULT-020 | FGR-REQ-002 | both timestamps = 0 | NOT stale, NORMAL |
