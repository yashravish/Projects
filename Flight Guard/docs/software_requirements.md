# FlightGuard Software Requirements

**Document ID:** FG-SRS-001  
**Version:** 1.0  
**Status:** Approved  
**Project:** FlightGuard Avionics Sensor Health Monitor  

> **Disclaimer:** This document is part of a portfolio project inspired by
> avionics software verification practices. It does not claim formal DO-178C
> certification or compliance with any regulatory standard.

---

## 1. Purpose

This document defines the software requirements for the FlightGuard Sensor
Health Monitor module. Each requirement is specific, testable, and traceable
to implementation and verification activities.

## 2. Scope

The FlightGuard module evaluates simulated aircraft sensor data and outputs:
- A **system operating mode**: NORMAL, DEGRADED, or FAILSAFE
- A set of **warning flags** indicating specific detected conditions

## 3. Definitions

| Term | Definition |
|------|-----------|
| Primary Airspeed | The main airspeed sensor reading |
| Redundant Airspeed | A secondary airspeed sensor reading for cross-check |
| Sensor Health Warning | A warning related to sensor data quality (INVALID_RANGE, DATA_STALE, SENSOR_DISAGREEMENT) |
| Flight Condition Warning | A warning related to flight state (OVERSPEED, STALL_RISK) |
| Stale Data | Sensor data whose age exceeds the defined threshold |

## 4. Engineering Assumptions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Valid airspeed range | 0.0 – 500.0 knots | Covers subsonic to high-speed aircraft |
| Valid altitude range | -1000.0 – 60000.0 ft | Below sea level to high-altitude operations |
| Valid AoA range | -10.0 – 40.0 degrees | Normal flight envelope |
| Stale data threshold | 2000 ms | 2-second timeout for sensor updates |
| Disagreement tolerance | 15.0 knots | Maximum acceptable difference between sensors |
| Overspeed threshold | 340.0 knots | Approximate VMO for demonstration |
| Stall risk AoA threshold | 15.0 degrees | Approximate critical AoA for demonstration |
| FAILSAFE warning count | 2 | Number of sensor health warnings triggering FAILSAFE |

## 5. Functional Requirements

### 5.1 Range Validation

**FGR-REQ-001:** The module shall set the INVALID_RANGE warning when the
primary airspeed value is outside the range [0.0, 500.0] knots (inclusive).

**FGR-REQ-010:** The module shall set the INVALID_RANGE warning when the
altitude value is outside the range [-1000.0, 60000.0] feet (inclusive).

**FGR-REQ-011:** The module shall set the INVALID_RANGE warning when the
angle of attack value is outside the range [-10.0, 40.0] degrees (inclusive).

### 5.2 Stale Data Detection

**FGR-REQ-002:** The module shall set the DATA_STALE warning when the age
of sensor data (current_time_ms - sensor_timestamp_ms) exceeds 2000
milliseconds.

> Note: Age exactly equal to the threshold (2000 ms) is NOT considered stale.
> Only values strictly greater than 2000 ms trigger this warning.

**FGR-REQ-002a:** If current_time_ms is less than sensor_timestamp_ms
(clock anomaly), the module shall treat the data as stale.

### 5.3 Sensor Validity

**FGR-REQ-003:** The module shall immediately enter FAILSAFE mode when the
sensor_valid flag is false, without evaluating other sensor data.

### 5.4 Overspeed Detection

**FGR-REQ-004:** The module shall set the OVERSPEED_WARNING when the primary
airspeed exceeds 340.0 knots.

> Note: Overspeed is only evaluated when sensor data is within valid range.

### 5.5 Stall Risk Detection

**FGR-REQ-005:** The module shall set the STALL_RISK warning when the angle
of attack exceeds 15.0 degrees.

> Note: Stall risk is only evaluated when sensor data is within valid range.

### 5.6 Sensor Disagreement Detection

**FGR-REQ-006:** The module shall set the SENSOR_DISAGREEMENT warning when
redundant airspeed data is available and the absolute difference between
primary and redundant airspeed values exceeds 15.0 knots.

> Note: This check is skipped when redundant_airspeed_available is false.

### 5.7 Mode Determination

**FGR-REQ-007:** The module shall enter DEGRADED mode when exactly one
sensor health warning (INVALID_RANGE, DATA_STALE, or SENSOR_DISAGREEMENT)
is active.

**FGR-REQ-008:** The module shall enter FAILSAFE mode when two or more
sensor health warnings are simultaneously active.

**FGR-REQ-009:** The module shall remain in NORMAL mode when no sensor
health warnings are active and sensor_valid is true.

### 5.8 Design Constraints

**FGR-REQ-012:** The module shall not use dynamic memory allocation.

**FGR-REQ-013:** The module shall produce deterministic output: identical
inputs shall always produce identical outputs.

**FGR-REQ-014:** Flight condition warnings (OVERSPEED, STALL_RISK) shall
not independently cause transition from NORMAL to DEGRADED or FAILSAFE mode.

**FGR-REQ-015:** The module shall return an error code when called with
NULL pointer arguments, without modifying any output.

## 6. Non-Functional Requirements

**FGR-NFR-001:** The module shall compile with GCC using `-std=c99 -Wall
-Wextra -Werror -pedantic` without warnings.

**FGR-NFR-002:** The module source code shall follow safety-critical coding
practices: small functions, explicit types, defensive checks, no hidden
side effects.

**FGR-NFR-003:** All requirements shall be traceable to implementation
functions and test cases.

## 7. Requirement Summary

| Requirement | Category | Description |
|-------------|----------|-------------|
| FGR-REQ-001 | Range | Airspeed out of range sets INVALID_RANGE |
| FGR-REQ-002 | Stale | Data age > 2000ms sets DATA_STALE |
| FGR-REQ-002a | Stale | Clock anomaly treated as stale |
| FGR-REQ-003 | Validity | sensor_valid=false triggers FAILSAFE |
| FGR-REQ-004 | Overspeed | Airspeed > 340kt sets OVERSPEED |
| FGR-REQ-005 | Stall | AoA > 15deg sets STALL_RISK |
| FGR-REQ-006 | Disagree | |primary-redundant| > 15kt sets DISAGREEMENT |
| FGR-REQ-007 | Mode | 1 health warning -> DEGRADED |
| FGR-REQ-008 | Mode | 2+ health warnings -> FAILSAFE |
| FGR-REQ-009 | Mode | 0 health warnings -> NORMAL |
| FGR-REQ-010 | Range | Altitude out of range sets INVALID_RANGE |
| FGR-REQ-011 | Range | AoA out of range sets INVALID_RANGE |
| FGR-REQ-012 | Constraint | No dynamic memory allocation |
| FGR-REQ-013 | Constraint | Deterministic behavior |
| FGR-REQ-014 | Mode | Flight warnings don't affect mode |
| FGR-REQ-015 | Defensive | NULL pointers return error |
