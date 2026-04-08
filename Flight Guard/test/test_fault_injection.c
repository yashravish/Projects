/**
 * @file test_fault_injection.c
 * @brief FlightGuard Fault Injection Tests
 *
 * Tests that specifically inject fault conditions into sensor inputs
 * to verify the module's ability to detect and respond to abnormal
 * data. Covers edge values at boundaries and combined faults.
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
 * Fault Injection Test Cases
 * ======================================================================== */

/**
 * TC-FAULT-001: Inject sensor_valid = false
 * Verifies immediate FAILSAFE on sensor invalidity.
 * Traces to: FGR-REQ-003
 */
static void test_fault_sensor_invalid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_valid = false;
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "FAILSAFE expected on sensor_valid=false");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-002: Inject negative airspeed
 * Verifies INVALID_RANGE detection for physically impossible airspeed.
 * Traces to: FGR-REQ-001
 */
static void test_fault_airspeed_negative(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = -50.0;
    input.redundant_airspeed_knots = -49.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for negative airspeed");
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Should be DEGRADED with one health warning");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-003: Inject extremely high airspeed
 * Verifies INVALID_RANGE for extreme out-of-range value.
 * Traces to: FGR-REQ-001
 */
static void test_fault_airspeed_extreme_high(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 99999.0;
    input.redundant_airspeed_knots = 99998.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for extreme airspeed");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-004: Inject very large timestamp age
 * Verifies DATA_STALE for very old sensor data.
 * Traces to: FGR-REQ-002
 */
static void test_fault_stale_large_age(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 0;
    input.current_time_ms     = 1000000;  /* 1,000 seconds old */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE for extremely old data");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-005: Inject large disagreement
 * Verifies SENSOR_DISAGREEMENT for large primary-redundant difference.
 * Traces to: FGR-REQ-006
 */
static void test_fault_disagreement_large(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 200.0;
    input.redundant_airspeed_knots = 350.0;  /* diff = 150 >> 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT for large disagreement");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-006: Boundary - airspeed just below minimum (edge)
 * Traces to: FGR-REQ-001
 */
static void test_fault_boundary_airspeed_just_below_min(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = -0.1;
    input.redundant_airspeed_knots = 0.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for airspeed just below 0.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-007: Boundary - airspeed just above maximum (edge)
 * Traces to: FGR-REQ-001
 */
static void test_fault_boundary_airspeed_just_above_max(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 500.1;
    input.redundant_airspeed_knots = 499.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for airspeed just above 500.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-008: Boundary - altitude just below minimum (edge)
 * Traces to: FGR-REQ-010
 */
static void test_fault_boundary_altitude_just_below_min(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = -1000.1;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for altitude just below -1000.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-009: Boundary - altitude just above maximum (edge)
 * Traces to: FGR-REQ-010
 */
static void test_fault_boundary_altitude_just_above_max(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = 60000.1;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for altitude just above 60000.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-010: Boundary - AoA just below minimum (edge)
 * Traces to: FGR-REQ-011
 */
static void test_fault_boundary_aoa_just_below_min(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = -10.1;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for AoA just below -10.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-011: Boundary - AoA just above maximum (edge)
 * Traces to: FGR-REQ-011
 */
static void test_fault_boundary_aoa_just_above_max(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 40.1;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE for AoA just above 40.0");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-012: Boundary - stale threshold exact (age == 2000, NOT stale)
 * Traces to: FGR-REQ-002
 */
static void test_fault_stale_boundary_exact(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 3000;  /* age = 2000 == threshold, NOT stale */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_DATA_STALE,
                         "Age exactly at threshold should NOT be stale");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-013: Boundary - disagreement exactly at tolerance (NOT triggered)
 * Traces to: FGR-REQ-006
 */
static void test_fault_disagree_boundary_exact(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 250.0;
    input.redundant_airspeed_knots = 235.0;  /* diff = 15.0 == tolerance */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                         "Diff exactly at tolerance should NOT trigger");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-014: Boundary - disagreement just over tolerance (triggered)
 * Traces to: FGR-REQ-006
 */
static void test_fault_disagree_boundary_over(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 250.0;
    input.redundant_airspeed_knots = 234.9;  /* diff = 15.1 > 15.0 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "Diff just over tolerance should trigger");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-015: Boundary - overspeed just above threshold
 * Traces to: FGR-REQ-004
 */
static void test_fault_overspeed_boundary_over(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 340.1;
    input.redundant_airspeed_knots = 340.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_OVERSPEED,
                       "Airspeed just above 340 should trigger overspeed");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-016: Boundary - stall risk just above threshold
 * Traces to: FGR-REQ-005
 */
static void test_fault_stall_boundary_over(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 15.1;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_STALL_RISK,
                       "AoA just above 15 should trigger stall risk");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-017: Combined fault - stale data AND sensor disagreement
 * Should trigger FAILSAFE (2 health warnings).
 * Traces to: FGR-REQ-002, FGR-REQ-006, FGR-REQ-008
 */
static void test_fault_combined_stale_and_disagree(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms      = 1000;
    input.current_time_ms          = 5000;   /* stale */
    input.redundant_airspeed_knots = 300.0;  /* disagreement */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "FAILSAFE expected with stale + disagreement");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT should be set");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-018: Combined fault - stale data AND invalid range
 * Should trigger FAILSAFE (2 health warnings).
 * Traces to: FGR-REQ-002, FGR-REQ-001, FGR-REQ-008
 */
static void test_fault_combined_stale_and_invalid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 5000;    /* stale */
    input.airspeed_knots      = 600.0;   /* out of range */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "FAILSAFE expected with stale + invalid range");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-019: Combined fault - invalid range AND disagreement
 * Should trigger FAILSAFE (2 health warnings).
 * Traces to: FGR-REQ-001, FGR-REQ-006, FGR-REQ-008
 */
static void test_fault_combined_invalid_and_disagree(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 600.0;   /* out of range */
    input.redundant_airspeed_knots = 200.0;   /* disagreement */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "FAILSAFE expected with invalid range + disagreement");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT should be set");
    FG_TEST_PASS();
}

/**
 * TC-FAULT-020: Sensor valid with zero timestamp values
 * Verifies behavior when both timestamps are zero (age = 0, fresh).
 * Traces to: FGR-REQ-002
 */
static void test_fault_zero_timestamps(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 0;
    input.current_time_ms     = 0;  /* age = 0 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_DATA_STALE,
                         "Zero-age data should not be stale");
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Should be NORMAL with zero-age data");
    FG_TEST_PASS();
}

/* ========================================================================
 * Test Suite Runner
 * ======================================================================== */

void run_fault_injection_tests(void)
{
    printf("\n--- Fault Injection Tests ---\n");

    FG_RUN_TEST(test_fault_sensor_invalid);
    FG_RUN_TEST(test_fault_airspeed_negative);
    FG_RUN_TEST(test_fault_airspeed_extreme_high);
    FG_RUN_TEST(test_fault_stale_large_age);
    FG_RUN_TEST(test_fault_disagreement_large);
    FG_RUN_TEST(test_fault_boundary_airspeed_just_below_min);
    FG_RUN_TEST(test_fault_boundary_airspeed_just_above_max);
    FG_RUN_TEST(test_fault_boundary_altitude_just_below_min);
    FG_RUN_TEST(test_fault_boundary_altitude_just_above_max);
    FG_RUN_TEST(test_fault_boundary_aoa_just_below_min);
    FG_RUN_TEST(test_fault_boundary_aoa_just_above_max);
    FG_RUN_TEST(test_fault_stale_boundary_exact);
    FG_RUN_TEST(test_fault_disagree_boundary_exact);
    FG_RUN_TEST(test_fault_disagree_boundary_over);
    FG_RUN_TEST(test_fault_overspeed_boundary_over);
    FG_RUN_TEST(test_fault_stall_boundary_over);
    FG_RUN_TEST(test_fault_combined_stale_and_disagree);
    FG_RUN_TEST(test_fault_combined_stale_and_invalid);
    FG_RUN_TEST(test_fault_combined_invalid_and_disagree);
    FG_RUN_TEST(test_fault_zero_timestamps);
}
