# FlightGuard: Avionics Sensor Health Monitor + Verification Bench

A deterministic C module that evaluates simulated aircraft sensor data and determines system operating mode (NORMAL, DEGRADED, FAILSAFE) with detailed warning flags. Built as a portfolio project to demonstrate strong alignment with **avionics software verification** practices.

> **Disclaimer:** This is a portfolio project inspired by avionics verification practices. It does **not** claim formal DO-178C certification or compliance with any regulatory standard.

---

## Why This Project

This project demonstrates competencies directly relevant to **Avionics Software Verification Engineer** roles:

| Competency | How Demonstrated |
|------------|-----------------|
| **Embedded C Development** | Deterministic C99 module with no dynamic allocation, no global state |
| **Requirements-Based Testing** | 16 numbered requirements, each with traceable test cases |
| **Verification Discipline** | 66 tests across unit, integration, and fault injection levels |
| **Safety-Critical Coding** | Small functions, explicit types, defensive handling, strict warnings |
| **Traceability** | Bidirectional requirement → code → test matrix |
| **Fault Injection** | 20 dedicated fault injection tests with boundary analysis |
| **HIL-Style Validation** | Scenario replay simulator with mode transition visibility |
| **Documentation** | 7 formal verification documents (SRS, LLD, VP, TC, TM, CRC, VS) |
| **CI/CD** | GitHub Actions pipeline for automated build and test |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  FG_SensorInput │────>│   fg_evaluate()  │────>│  FG_EvalResult   │
│   (8 fields)    │     │  (Stateless)     │     │  (mode+warnings) │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

**Inputs:** airspeed, altitude, angle of attack, sensor validity, timestamps, redundant airspeed

**Outputs:**
- **System Mode:** NORMAL → DEGRADED → FAILSAFE
- **Warning Flags:** OVERSPEED_WARNING, STALL_RISK, SENSOR_DISAGREEMENT, DATA_STALE, INVALID_RANGE

**Key Design Properties:**
- Stateless: no internal state between calls
- Deterministic: identical inputs → identical outputs
- No dynamic memory allocation
- No external dependencies beyond C99 standard library

---

## Project Structure

```
Flight Guard/
├── include/
│   └── flightguard.h          # Public API (types, constants, declarations)
├── src/
│   ├── flightguard.c          # Core evaluation logic
│   └── main.c                 # Demo entry point
├── test/
│   ├── test_harness.h         # Minimal test framework
│   ├── test_unit.c            # 40 unit tests
│   ├── test_integration.c     # 6 integration tests
│   ├── test_fault_injection.c # 20 fault injection tests
│   └── test_main.c            # Test runner
├── sim/
│   └── simulator.c            # HIL-style scenario replay
├── docs/
│   ├── software_requirements.md
│   ├── low_level_design.md
│   ├── verification_plan.md
│   ├── test_cases.md
│   ├── traceability_matrix.md
│   ├── code_review_checklist.md
│   └── verification_summary.md
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI
├── Makefile                   # GNU Make build system
├── build.ps1                  # PowerShell build script (Windows)
├── .gitignore
└── README.md
```

---

## Build Instructions

### Prerequisites

- **GCC** (any recent version supporting C99)
- **GNU Make** (Linux/macOS) or **PowerShell** (Windows)

### Linux / macOS (Make)

```bash
# Build the demo executable
make

# Build and run the demo
make run

# Build and run all tests
make test

# Build and run the HIL simulator
make sim

# Generate code coverage
make coverage

# Clean build artifacts
make clean
```

### Windows (PowerShell)

```powershell
# Build the demo executable
.\build.ps1

# Build and run the demo
.\build.ps1 run

# Build and run all tests
.\build.ps1 test

# Build and run the HIL simulator
.\build.ps1 sim

# Build all executables
.\build.ps1 all

# Generate code coverage
.\build.ps1 coverage

# Clean build artifacts
.\build.ps1 clean
```

### Manual GCC Commands

If neither Make nor PowerShell scripts work for your environment:

```bash
# Create build directory
mkdir build

# Build demo
gcc -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude -o build/flightguard_demo src/flightguard.c src/main.c

# Build tests
gcc -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude -Itest -o build/flightguard_tests src/flightguard.c test/test_main.c test/test_unit.c test/test_integration.c test/test_fault_injection.c

# Build simulator
gcc -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude -o build/flightguard_sim src/flightguard.c sim/simulator.c
```

---

## Running the Project

### Demo

```
$ ./build/flightguard_demo

============================================================
  FlightGuard: Avionics Sensor Health Monitor v1.0.0
============================================================

--- Demo: Nominal Cruise Evaluation ---

Input:
  Airspeed (primary):   250.0 knots
  Altitude:             35000.0 ft
  Angle of Attack:      3.5 deg
  Sensor Valid:         YES
  ...

Result:
  System Mode: NORMAL
  Warnings:    NONE

--- Demo: Degraded Scenario (Stale Data) ---
...
Result:
  System Mode: DEGRADED
  Warnings:    DATA_STALE

--- Demo: FAILSAFE Scenario (Sensor Invalid) ---
...
Result:
  System Mode: FAILSAFE
  Warnings:    NONE
```

### Tests

```
$ ./build/flightguard_tests

============================================================
  FlightGuard Test Suite
============================================================

--- Unit Tests ---
  [PASS] test_nominal_all_valid
  [PASS] test_null_input
  [PASS] test_null_result
  ...

--- Integration Tests ---
  [PASS] test_integ_nominal_climb
  ...

--- Fault Injection Tests ---
  [PASS] test_fault_sensor_invalid
  ...

========================================
  Test Results Summary
========================================
  Total:  66
  Passed: 66
  Failed: 0
========================================
  STATUS: ALL TESTS PASSED
========================================
```

### Simulator

```
$ ./build/flightguard_sim

============================================================
  FlightGuard HIL-Style Scenario Simulator v1.0.0
============================================================
  Replaying 3 built-in scenarios...

------------------------------------------------------------
  SCENARIO: Nominal Flight Profile
  Complete flight from takeoff through cruise to landing...
------------------------------------------------------------

  [Step  1] Takeoff Roll                     | T=   1100 ms
            AS=80 kt  ALT=0 ft  AoA=2.0 deg  Valid=Y  Age=100 ms
            >> Mode: NORMAL      Warnings: NONE

  [Step  2] Rotation & Liftoff               | T=   2100 ms
            ...
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Software Requirements](docs/software_requirements.md) | 16 numbered, testable requirements |
| [Low-Level Design](docs/low_level_design.md) | Data structures, functions, control flow |
| [Verification Plan](docs/verification_plan.md) | Test strategy, levels, pass/fail criteria |
| [Test Cases](docs/test_cases.md) | Formal test case specifications |
| [Traceability Matrix](docs/traceability_matrix.md) | Requirement ↔ Code ↔ Test mapping |
| [Code Review Checklist](docs/code_review_checklist.md) | 11-category review guide |
| [Verification Summary](docs/verification_summary.md) | Results, coverage, limitations |

---

## Engineering Assumptions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Valid airspeed | 0–500 knots | Subsonic to high-speed envelope |
| Valid altitude | -1000–60000 ft | Below sea level to high-altitude |
| Valid AoA | -10–40 degrees | Normal flight envelope |
| Stale threshold | 2000 ms | 2-second sensor update timeout |
| Disagreement tolerance | 15 knots | Cross-check tolerance |
| Overspeed threshold | 340 knots | Approximate VMO |
| Stall risk threshold | 15° AoA | Approximate critical AoA |

---

## Limitations

- **Not certified:** Portfolio demonstration only; no regulatory compliance claimed
- **Generic thresholds:** Values are for demonstration, not aircraft-specific
- **Stateless module:** No mode transition debouncing or hysteresis
- **Single-threaded:** No concurrency or interrupt handling
- **No hardware interface:** All sensor data is simulated
- **No NaN explicit guard:** Handled implicitly by IEEE 754 range comparisons

---

## Future Improvements

- Mode transition hysteresis and debouncing
- Configurable runtime thresholds
- CSV scenario file import for simulator
- Additional sensor channels (vertical speed, heading, engine data)
- MISRA C:2012 compliance analysis
- Triple-redundancy voting logic
- Structured flight data logging
- Formal verification with model checking tools

---

## License

This project is provided for portfolio demonstration purposes.
