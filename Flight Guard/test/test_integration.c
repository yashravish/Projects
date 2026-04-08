/**
 * @file test_integration.c
 * @brief FlightGuard Integration Tests
 *
 * Validates end-to-end behavior across realistic multi-step flight
 * scenarios. Each test simulates a sequence of sensor inputs representing
 * a phase of flight and verifies correct mode transitions.
 */

#include "flightguard.h"
#include "test_harness.h"

/* ========================================================================
 * Helper: Create a nominal (healthy) sensor input
 * ======================================================================== */

static FG_SensorInput make_nominal(void)
{
    FG_SensorInput input;
    input.airspeed_knots             = 250.0;
    input.altitude_ft                = 35000.0;
    input.angle_of_attack_deg        = 3.5;
    input.sensor_valid               = true;
    input.sensor_timestamp_ms        = 1000;
    input.current_time_ms            = 1500;
    input.redundant_airspeed_available = true;
    input.redundant_airspeed_knots   = 252.0;
    return input;
}

/* ========================================================================
 * Integration Test Cases
 * ======================================================================== */

/**
 * TC-INTEG-001: Nominal climb from takeoff to cruise
 *
 * Scenario: Aircraft takes off, climbs, and reaches cruise altitude.
 * All sensor data remains valid throughout.
 *
 * Expected: NORMAL mode at every step, no warnings.
 * Traces to: FGR-REQ-009
 */
static void test_integ_nominal_climb(void)
{
    FG_EvalResult result;
    FG_SensorInput steps[] = {
        /* Takeoff roll */
        { .airspeed_knots = 80.0,   .altitude_ft = 0.0,     .angle_of_attack_deg = 2.0,
          .sensor_valid = true, .sensor_timestamp_ms = 100, .current_time_ms = 200,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 81.0 },
        /* Rotation */
        { .airspeed_knots = 150.0,  .altitude_ft = 50.0,    .angle_of_attack_deg = 8.0,
          .sensor_valid = true, .sensor_timestamp_ms = 500, .current_time_ms = 600,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 151.0 },
        /* Initial climb */
        { .airspeed_knots = 200.0,  .altitude_ft = 5000.0,  .angle_of_attack_deg = 5.0,
          .sensor_valid = true, .sensor_timestamp_ms = 2000, .current_time_ms = 2100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 201.0 },
        /* Climb through FL180 */
        { .airspeed_knots = 280.0,  .altitude_ft = 18000.0, .angle_of_attack_deg = 4.0,
          .sensor_valid = true, .sensor_timestamp_ms = 5000, .current_time_ms = 5100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 281.0 },
        /* Cruise at FL350 */
        { .airspeed_knots = 250.0,  .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 10000, .current_time_ms = 10100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    };

    int num_steps = (int)(sizeof(steps) / sizeof(steps[0]));

    for (int i = 0; i < num_steps; i++) {
        int rc = fg_evaluate(&steps[i], &result);
        FG_ASSERT_EQ_INT(rc, FG_OK, "Evaluation should succeed");
        FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                         "Mode should be NORMAL during nominal climb");
        FG_ASSERT_EQ_INT((int)result.warnings, (int)FG_WARN_NONE,
                         "No warnings during nominal climb");
    }

    FG_TEST_PASS();
}

/**
 * TC-INTEG-002: Stale sensor update during cruise
 *
 * Scenario: Aircraft is cruising normally, then sensor data stops
 * updating (timestamp becomes stale). After sensor recovers, mode
 * returns to NORMAL.
 *
 * Expected: NORMAL -> DEGRADED (stale) -> NORMAL (recovered)
 * Traces to: FGR-REQ-002, FGR-REQ-007, FGR-REQ-009
 */
static void test_integ_cruise_stale_sensor(void)
{
    FG_EvalResult result;

    /* Step 1: Normal cruise */
    FG_SensorInput input = make_nominal();
    input.sensor_timestamp_ms = 10000;
    input.current_time_ms     = 10500;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 1: should be NORMAL");

    /* Step 2: Sensor freezes - timestamp doesn't update */
    input.current_time_ms = 13000;  /* age = 3000ms > 2000ms */

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Step 2: should be DEGRADED (stale data)");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "Step 2: DATA_STALE should be set");

    /* Step 3: Sensor recovers - new timestamp */
    input.sensor_timestamp_ms = 13000;
    input.current_time_ms     = 13500;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 3: should be NORMAL after recovery");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_DATA_STALE,
                         "Step 3: DATA_STALE should be cleared");

    FG_TEST_PASS();
}

/**
 * TC-INTEG-003: Disagreement between redundant sensors
 *
 * Scenario: During cruise, the redundant airspeed sensor begins
 * drifting. Initially within tolerance, then exceeds threshold,
 * causing DEGRADED mode.
 *
 * Expected: NORMAL -> NORMAL (within tolerance) -> DEGRADED (disagreement)
 * Traces to: FGR-REQ-006, FGR-REQ-007
 */
static void test_integ_redundant_disagreement(void)
{
    FG_EvalResult result;
    FG_SensorInput input = make_nominal();

    /* Step 1: Normal, redundant agrees */
    input.redundant_airspeed_knots = 252.0;  /* diff = 2 */
    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 1: should be NORMAL");

    /* Step 2: Redundant drifts but within tolerance */
    input.redundant_airspeed_knots = 263.0;  /* diff = 13 <= 15 */
    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 2: should still be NORMAL");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                         "Step 2: no disagreement yet");

    /* Step 3: Redundant exceeds tolerance */
    input.redundant_airspeed_knots = 270.0;  /* diff = 20 > 15 */
    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Step 3: should be DEGRADED");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "Step 3: SENSOR_DISAGREEMENT should be set");

    FG_TEST_PASS();
}

/**
 * TC-INTEG-004: Invalid sensor data during approach
 *
 * Scenario: Aircraft is on approach with decreasing altitude. Altitude
 * sensor suddenly reports an obviously invalid value, then the sensor
 * is marked invalid entirely.
 *
 * Expected: NORMAL -> DEGRADED (invalid range) -> FAILSAFE (sensor invalid)
 * Traces to: FGR-REQ-010, FGR-REQ-007, FGR-REQ-003
 */
static void test_integ_invalid_sensor_approach(void)
{
    FG_EvalResult result;
    FG_SensorInput input = make_nominal();

    /* Step 1: Normal approach */
    input.airspeed_knots           = 140.0;
    input.altitude_ft              = 3000.0;
    input.angle_of_attack_deg      = 5.0;
    input.redundant_airspeed_knots = 141.0;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 1: should be NORMAL on approach");

    /* Step 2: Altitude sensor glitch - reports impossible value */
    input.altitude_ft = -5000.0;  /* below -1000 minimum */

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Step 2: should be DEGRADED (invalid range)");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "Step 2: INVALID_RANGE should be set");

    /* Step 3: Sensor subsystem fails completely */
    input.sensor_valid = false;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "Step 3: should be FAILSAFE (sensor invalid)");

    FG_TEST_PASS();
}

/**
 * TC-INTEG-005: Complete sensor invalidation causing FAILSAFE
 *
 * Scenario: Multiple faults accumulate (stale + disagreement) leading
 * to FAILSAFE mode through the dual-warning threshold.
 *
 * Expected: NORMAL -> DEGRADED (one fault) -> FAILSAFE (two faults)
 * Traces to: FGR-REQ-002, FGR-REQ-006, FGR-REQ-007, FGR-REQ-008
 */
static void test_integ_complete_sensor_failure(void)
{
    FG_EvalResult result;
    FG_SensorInput input = make_nominal();

    /* Step 1: Normal flight */
    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL, "Step 1: NORMAL");

    /* Step 2: Data goes stale (single fault -> DEGRADED) */
    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 5000;  /* age = 4000ms */

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Step 2: DEGRADED from stale data");

    /* Step 3: Redundant also disagrees (dual fault -> FAILSAFE) */
    input.redundant_airspeed_knots = 300.0;  /* diff = 50 > 15 */

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "Step 3: FAILSAFE from dual faults");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "Step 3: DATA_STALE present");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "Step 3: SENSOR_DISAGREEMENT present");

    FG_TEST_PASS();
}

/**
 * TC-INTEG-006: Overspeed during descent does not cause mode change
 *
 * Scenario: Aircraft accelerates during descent, briefly exceeding
 * overspeed threshold. Mode should remain NORMAL since overspeed is
 * a flight condition warning, not a sensor health fault.
 *
 * Expected: NORMAL throughout, OVERSPEED flag set during exceedance
 * Traces to: FGR-REQ-004, FGR-REQ-014
 */
static void test_integ_overspeed_descent(void)
{
    FG_EvalResult result;
    FG_SensorInput input = make_nominal();

    /* Step 1: Normal descent */
    input.airspeed_knots           = 300.0;
    input.altitude_ft              = 20000.0;
    input.redundant_airspeed_knots = 301.0;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 1: NORMAL during descent");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_OVERSPEED,
                         "Step 1: No overspeed");

    /* Step 2: Speed increases past threshold */
    input.airspeed_knots           = 345.0;
    input.redundant_airspeed_knots = 346.0;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 2: still NORMAL despite overspeed");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_OVERSPEED,
                       "Step 2: OVERSPEED flag set");

    /* Step 3: Speed reduces */
    input.airspeed_knots           = 310.0;
    input.redundant_airspeed_knots = 311.0;

    fg_evaluate(&input, &result);
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Step 3: NORMAL again");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_OVERSPEED,
                         "Step 3: OVERSPEED cleared");

    FG_TEST_PASS();
}

/* ========================================================================
 * Test Suite Runner
 * ======================================================================== */

void run_integration_tests(void)
{
    printf("\n--- Integration Tests ---\n");

    FG_RUN_TEST(test_integ_nominal_climb);
    FG_RUN_TEST(test_integ_cruise_stale_sensor);
    FG_RUN_TEST(test_integ_redundant_disagreement);
    FG_RUN_TEST(test_integ_invalid_sensor_approach);
    FG_RUN_TEST(test_integ_complete_sensor_failure);
    FG_RUN_TEST(test_integ_overspeed_descent);
}
