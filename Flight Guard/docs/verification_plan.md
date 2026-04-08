# FlightGuard Verification Plan

**Document ID:** FG-VP-001  
**Version:** 1.0  
**Project:** FlightGuard Avionics Sensor Health Monitor  

---

## 1. Purpose

This document defines the verification strategy for the FlightGuard module.
It describes the testing levels, methodologies, pass/fail criteria, and
regression approach used to demonstrate requirements coverage.

## 2. Verification Objectives

1. Demonstrate that all functional requirements (FGR-REQ-*) are satisfied
2. Verify correct behavior at boundary values
3. Validate module resilience to injected faults
4. Verify end-to-end behavior through realistic scenarios
5. Ensure deterministic and repeatable test execution

## 3. Test Levels

### 3.1 Unit Testing

**Scope:** Individual check functions and mode determination logic.

**Approach:**
- Each internal function is exercised through the public `fg_evaluate()` API
- Tests target specific requirements with controlled inputs
- Nominal, boundary, and error cases are covered for each check type

**Key test categories:**
- Range validation (airspeed, altitude, AoA)
- Stale data detection
- Sensor disagreement detection
- Overspeed detection
- Stall risk detection
- Mode determination rules
- Utility functions (string conversion, NULL handling)

**Test file:** `test/test_unit.c`

### 3.2 Integration Testing

**Scope:** Multi-step flight scenarios exercising the module through
realistic sequences of inputs.

**Approach:**
- Each test simulates a sequence of evaluation calls representing a
  flight phase (e.g., takeoff → cruise → approach)
- Mode transitions are verified at each step
- Tests validate that the module correctly tracks changing conditions

**Scenarios:**
1. Nominal flight from takeoff through landing
2. Stale sensor during cruise with recovery
3. Drifting redundant sensor causing disagreement
4. Invalid sensor data during approach
5. Progressive sensor failure escalating to FAILSAFE
6. Overspeed during descent (mode should not change)

**Test file:** `test/test_integration.c`

### 3.3 Fault Injection Testing

**Scope:** Deliberate injection of abnormal inputs to verify fault detection.

**Approach:**
- Inject specific fault conditions into otherwise nominal inputs
- Verify correct warning flags and mode transitions
- Focus on edge values at threshold boundaries
- Test combined fault scenarios

**Fault types injected:**
- Invalid sensor_valid flag
- Negative airspeed
- Extremely high airspeed
- Very large timestamp age
- Large primary/redundant disagreement
- Values just inside/outside each threshold boundary
- Combined faults (stale + disagreement, stale + invalid, invalid + disagreement)
- Zero-value edge cases

**Test file:** `test/test_fault_injection.c`

## 4. Boundary Value Analysis

The following boundaries are explicitly tested:

| Parameter | Below Range | At Min | At Max | Above Range |
|-----------|------------|--------|--------|-------------|
| Airspeed | -0.1, -1.0, -50.0 | 0.0 | 500.0 | 500.1, 501.0, 99999.0 |
| Altitude | -1000.1 | -1000.0 | 60000.0 | 60000.1 |
| AoA | -10.1 | -10.0 (implicit) | 40.0 (implicit) | 40.1 |
| Stale age | 2000 ms (not stale) | — | — | 2001 ms (stale) |
| Disagreement | 15.0 kt (not triggered) | — | — | 15.1 kt (triggered) |
| Overspeed | 340.0 kt (not triggered) | — | — | 340.1 kt (triggered) |
| Stall AoA | 15.0 deg (not triggered) | — | — | 15.1 deg (triggered) |

## 5. Pass/Fail Criteria

### 5.1 Individual Test Pass Criteria
- All assertions within the test pass
- Return code from `fg_evaluate()` matches expected value
- System mode matches expected mode
- Warning flags match expected bitmask (set and clear checks)

### 5.2 Suite Pass Criteria
- All tests pass (zero failures)
- Test executable returns exit code 0
- No undefined behavior detected
- No memory leaks (verified by design: no dynamic allocation)

### 5.3 Failure Handling
- Failed assertions print file, line number, and descriptive message
- Test execution continues after individual test failure
- Final summary reports total/pass/fail counts
- Non-zero exit code enables CI/CD failure detection

## 6. Test Infrastructure

### 6.1 Test Harness
A custom minimal harness (`test/test_harness.h`) provides:
- `FG_ASSERT(condition, message)` — general assertion
- `FG_ASSERT_EQ_INT(actual, expected, message)` — integer comparison
- `FG_ASSERT_FLAG_SET(warnings, flag, message)` — bitmask set check
- `FG_ASSERT_FLAG_CLEAR(warnings, flag, message)` — bitmask clear check
- `FG_RUN_TEST(function)` — test runner with pass/fail reporting
- `FG_TEST_PASS()` — mark test as passed

### 6.2 Rationale
A custom harness is used instead of an external framework to:
- Minimize dependencies (no library installation needed)
- Ensure compilation on any system with a C99 compiler
- Maintain simplicity aligned with safety-critical practices

## 7. Coverage Strategy

### 7.1 Requirements Coverage
Every requirement (FGR-REQ-001 through FGR-REQ-015) has at least one
dedicated test case. The traceability matrix (see `traceability_matrix.md`)
maps each requirement to its test case(s).

### 7.2 Code Coverage
- GCC's `--coverage` flag enables gcov instrumentation
- `make coverage` builds instrumented test binary and generates `.gcov` files
- Line-by-line coverage is inspected in the coverage output

### 7.3 Coverage Targets
- Statement coverage: target > 95%
- Branch coverage: all decision branches exercised
- Boundary values: explicitly tested for all thresholds

## 8. Regression Strategy

### 8.1 Approach
- All tests are executed on every build (`make test`)
- CI pipeline (GitHub Actions) runs full test suite on push/PR
- Any new defect fix requires a corresponding regression test
- Test suite is designed to be fast (< 1 second execution)

### 8.2 Version Control
- Test source files are version-controlled alongside production code
- Test additions are reviewed alongside code changes
- No test may be removed without documented justification

## 9. HIL-Style Simulation

The simulator (`sim/simulator.c`) provides an additional verification
layer by replaying multi-step scenarios with detailed output.

**Capabilities:**
- Three built-in scenarios covering nominal, fault, and FAILSAFE conditions
- Timestamped output showing inputs and evaluation results
- Mode transition visibility across scenario steps

**Note:** This is not true hardware-in-the-loop testing. It demonstrates
the concept of scenario-based validation in a portable software environment.

## 10. Verification Environment

| Component | Specification |
|-----------|--------------|
| Compiler | GCC (C99 mode) |
| Warning flags | -Wall -Wextra -Werror -pedantic |
| Target OS | Linux (CI), Windows (development) |
| Test execution | Automated via make/PowerShell |
| CI platform | GitHub Actions (ubuntu-latest) |
