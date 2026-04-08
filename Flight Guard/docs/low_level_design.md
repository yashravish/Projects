# FlightGuard Low-Level Design

**Document ID:** FG-LLD-001  
**Version:** 1.0  
**Project:** FlightGuard Avionics Sensor Health Monitor  

---

## 1. Overview

This document describes the low-level software design of the FlightGuard
Sensor Health Monitor module. It covers data structures, function
responsibilities, control flow, and key design choices aligned with
safety-critical software practices.

## 2. Architecture

The FlightGuard module is a stateless, single-function evaluation engine.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  FG_SensorInput │────>│   fg_evaluate()  │────>│  FG_EvalResult   │
│  (Caller Data)  │     │  (Core Logic)    │     │  (Mode+Warnings) │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

**Key architectural property:** The module maintains no internal state.
Every call to `fg_evaluate()` produces output solely from the provided
input. This guarantees determinism and simplifies verification.

## 3. Data Structures

### 3.1 FG_SensorInput

Input structure provided by the caller for each evaluation cycle.

| Field | Type | Description |
|-------|------|-------------|
| `airspeed_knots` | `double` | Primary airspeed reading (knots) |
| `altitude_ft` | `double` | Altitude reading (feet) |
| `angle_of_attack_deg` | `double` | Angle of attack (degrees) |
| `sensor_valid` | `bool` | Master sensor validity flag |
| `sensor_timestamp_ms` | `uint64_t` | Timestamp of sensor data (ms) |
| `current_time_ms` | `uint64_t` | Current system time (ms) |
| `redundant_airspeed_available` | `bool` | Is redundant airspeed present? |
| `redundant_airspeed_knots` | `double` | Redundant airspeed reading (knots) |

**Design notes:**
- `uint64_t` for timestamps provides sufficient range for long-duration flights
- `bool` flags use `stdbool.h` for C99 compliance
- All fields are value types; no pointers in the input structure

### 3.2 FG_EvalResult

Output structure populated by `fg_evaluate()`.

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `FG_SystemMode` | Determined operating mode |
| `warnings` | `uint32_t` | Bitmask of active warning flags |

### 3.3 FG_SystemMode (Enum)

| Value | Integer | Description |
|-------|---------|-------------|
| `FG_MODE_NORMAL` | 0 | All sensors healthy |
| `FG_MODE_DEGRADED` | 1 | One sensor health fault |
| `FG_MODE_FAILSAFE` | 2 | Sensor invalid or multiple faults |

### 3.4 FG_WarningFlag (Enum, Bitmask)

| Flag | Value | Category | Description |
|------|-------|----------|-------------|
| `FG_WARN_NONE` | 0x00 | — | No warnings |
| `FG_WARN_OVERSPEED` | 0x01 | Flight | Airspeed exceeds threshold |
| `FG_WARN_STALL_RISK` | 0x02 | Flight | AoA exceeds threshold |
| `FG_WARN_SENSOR_DISAGREEMENT` | 0x04 | Health | Primary/redundant mismatch |
| `FG_WARN_DATA_STALE` | 0x08 | Health | Sensor data too old |
| `FG_WARN_INVALID_RANGE` | 0x10 | Health | Sensor value out of bounds |

**Bitmask design:** Using power-of-two values allows multiple simultaneous
warnings to be stored efficiently in a single `uint32_t` field, tested with
bitwise AND operations.

## 4. Function Responsibilities

### 4.1 Public Functions

#### `fg_evaluate(const FG_SensorInput *input, FG_EvalResult *result)`
- **Purpose:** Primary entry point; evaluates sensor data
- **Inputs:** Pointer to sensor data, pointer to result buffer
- **Output:** Populates result with mode and warnings; returns status code
- **Preconditions:** Both pointers must be non-NULL
- **Postconditions:** Result is fully initialized; mode and warnings are consistent

#### `fg_mode_to_string(FG_SystemMode mode)`
- **Purpose:** Convert mode enum to human-readable string
- **Returns:** Static string literal (never NULL)

#### `fg_warnings_to_string(uint32_t warnings, char *buf, size_t buf_size)`
- **Purpose:** Format warning bitmask into readable text
- **Returns:** Number of characters written, or -1 on error

### 4.2 Internal Functions (Static)

All internal functions are declared `static` to limit visibility.

| Function | Responsibility | Traces To |
|----------|---------------|-----------|
| `fg_is_airspeed_in_range()` | Validate airspeed bounds | FGR-REQ-001 |
| `fg_is_altitude_in_range()` | Validate altitude bounds | FGR-REQ-010 |
| `fg_is_aoa_in_range()` | Validate AoA bounds | FGR-REQ-011 |
| `fg_are_ranges_valid()` | Aggregate range validation | FGR-REQ-001, 010, 011 |
| `fg_is_data_stale()` | Check sensor data age | FGR-REQ-002, 002a |
| `fg_is_disagreement_detected()` | Compare primary vs redundant | FGR-REQ-006 |
| `fg_is_overspeed()` | Check overspeed threshold | FGR-REQ-004 |
| `fg_is_stall_risk()` | Check stall AoA threshold | FGR-REQ-005 |
| `fg_popcount()` | Count set bits in bitmask | Support for FGR-REQ-007/008 |
| `fg_determine_mode()` | Apply mode determination rules | FGR-REQ-003, 007, 008, 009 |

## 5. Control Flow

The `fg_evaluate()` function follows a strict, sequential evaluation order:

```
1. Null pointer check → return FG_ERR_NULL if failed
2. Initialize result to NORMAL/NONE
3. Check sensor_valid
   └─ If false → set FAILSAFE, return immediately
4. Check range validity
   └─ If any value out-of-range → set INVALID_RANGE
5. Check data staleness
   └─ If age > threshold or clock anomaly → set DATA_STALE
6. Check redundant disagreement
   └─ If available and diff > tolerance → set SENSOR_DISAGREEMENT
7. If INVALID_RANGE is NOT set:
   a. Check overspeed → set OVERSPEED if applicable
   b. Check stall risk → set STALL_RISK if applicable
8. Determine mode from sensor_valid and health warnings
9. Return FG_OK
```

**Critical design decision (Step 7):** Flight condition warnings are only
evaluated when range data is valid. If INVALID_RANGE is set, the airspeed
and AoA values cannot be trusted, so overspeed and stall checks are skipped.
This prevents false alarms from corrupted data.

## 6. Mode Determination Logic

```
┌──────────────────────────────┐
│    Is sensor_valid false?    │──YES──> FAILSAFE
└──────────┬───────────────────┘
           │ NO
┌──────────▼───────────────────┐
│ Count sensor health warnings │
│ (INVALID_RANGE + DATA_STALE  │
│  + SENSOR_DISAGREEMENT)      │
└──────────┬───────────────────┘
           │
    ┌──────▼──────┐
    │ count >= 2? │──YES──> FAILSAFE
    └──────┬──────┘
           │ NO
    ┌──────▼──────┐
    │ count == 1? │──YES──> DEGRADED
    └──────┬──────┘
           │ NO
           └──────────────────> NORMAL
```

## 7. Safety-Minded Design Choices

### 7.1 No Dynamic Memory
All data structures are stack-allocated or statically defined. The module
never calls `malloc()`, `calloc()`, `realloc()`, or `free()`.

### 7.2 No Global Mutable State
The module has no global variables. All state is passed through function
parameters. This eliminates race conditions and initialization ordering
issues.

### 7.3 Defensive Input Handling
- NULL pointer arguments are checked before any processing
- Clock anomalies (negative age) are treated conservatively as stale data
- Out-of-range values suppress downstream flight condition checks
- Redundant airspeed is only checked when explicitly marked as available

### 7.4 Determinism
No random number generation, no time-dependent behavior (time is an input),
no floating-point rounding mode dependencies. The same input always
produces the same output.

### 7.5 Predictable Control Flow
No recursion, no function pointers, no complex branching. Each check is
a simple conditional with clear true/false paths.

### 7.6 Minimal Dependencies
The module depends only on C99 standard library headers:
- `stdbool.h` — boolean type
- `stdint.h` — fixed-width integers
- `stddef.h` — `size_t`
- `stdio.h` — `snprintf()` for string formatting
- `string.h` — not used in core logic

No external libraries are required.

## 8. File Structure

```
include/
  flightguard.h      Public API header (types, constants, function declarations)

src/
  flightguard.c      Core evaluation logic (all static helpers + public API)
  main.c             Demo entry point

test/
  test_harness.h     Minimal test framework macros
  test_unit.c        Unit tests
  test_integration.c Integration tests
  test_fault_injection.c  Fault injection tests
  test_main.c        Test runner

sim/
  simulator.c        HIL-style scenario replay
```
