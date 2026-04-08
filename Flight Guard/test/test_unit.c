/**
 * @file test_unit.c
 * @brief FlightGuard Unit Tests
 *
 * Requirements-based unit tests covering individual checks, boundary
 * values, and mode determination logic. Each test traces to one or
 * more requirements from software_requirements.md.
 */

#include "flightguard.h"
#include "test_harness.h"
#include <string.h>

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
 * Test Cases
 * ======================================================================== */

/* TC-UNIT-001: Nominal evaluation produces NORMAL mode, no warnings
 * Traces to: FGR-REQ-009 */
static void test_nominal_all_valid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    int rc = fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(rc, FG_OK, "Return code should be FG_OK");
    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL, "Mode should be NORMAL");
    FG_ASSERT_EQ_INT(result.warnings, FG_WARN_NONE, "No warnings expected");
    FG_TEST_PASS();
}

/* TC-UNIT-002: NULL input pointer returns error
 * Traces to: FGR-REQ-015 */
static void test_null_input(void)
{
    FG_EvalResult result;
    int rc = fg_evaluate(NULL, &result);

    FG_ASSERT_EQ_INT(rc, FG_ERR_NULL, "Should return FG_ERR_NULL for NULL input");
    FG_TEST_PASS();
}

/* TC-UNIT-003: NULL result pointer returns error
 * Traces to: FGR-REQ-015 */
static void test_null_result(void)
{
    FG_SensorInput input = make_nominal();
    int rc = fg_evaluate(&input, NULL);

    FG_ASSERT_EQ_INT(rc, FG_ERR_NULL, "Should return FG_ERR_NULL for NULL result");
    FG_TEST_PASS();
}

/* TC-UNIT-004: Airspeed below valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-001 */
static void test_airspeed_below_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots = -1.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for airspeed < 0");
    FG_TEST_PASS();
}

/* TC-UNIT-005: Airspeed above valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-001 */
static void test_airspeed_above_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots = 501.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for airspeed > 500");
    FG_TEST_PASS();
}

/* TC-UNIT-006: Altitude below valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-010 */
static void test_altitude_below_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = -1001.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for altitude < -1000");
    FG_TEST_PASS();
}

/* TC-UNIT-007: Altitude above valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-010 */
static void test_altitude_above_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = 60001.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for altitude > 60000");
    FG_TEST_PASS();
}

/* TC-UNIT-008: AoA below valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-011 */
static void test_aoa_below_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = -11.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for AoA < -10");
    FG_TEST_PASS();
}

/* TC-UNIT-009: AoA above valid range sets INVALID_RANGE
 * Traces to: FGR-REQ-011 */
static void test_aoa_above_range(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 41.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set for AoA > 40");
    FG_TEST_PASS();
}

/* TC-UNIT-010: Fresh data does not set DATA_STALE
 * Traces to: FGR-REQ-002 */
static void test_data_fresh(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 2000;  /* age = 1000ms <= 2000ms threshold */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_DATA_STALE,
                         "DATA_STALE should NOT be set for fresh data");
    FG_TEST_PASS();
}

/* TC-UNIT-011: Stale data sets DATA_STALE
 * Traces to: FGR-REQ-002 */
static void test_data_stale(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 4000;  /* age = 3000ms > 2000ms threshold */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set for stale data");
    FG_TEST_PASS();
}

/* TC-UNIT-012: Boundary - age exactly at threshold is NOT stale
 * Traces to: FGR-REQ-002 */
static void test_data_stale_boundary_not_stale(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 3000;  /* age = 2000ms, NOT > 2000ms */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_DATA_STALE,
                         "DATA_STALE should NOT be set at exact threshold");
    FG_TEST_PASS();
}

/* TC-UNIT-013: Boundary - age one ms over threshold IS stale
 * Traces to: FGR-REQ-002 */
static void test_data_stale_boundary_stale(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 3001;  /* age = 2001ms > 2000ms */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set at threshold + 1");
    FG_TEST_PASS();
}

/* TC-UNIT-014: Clock anomaly (current < timestamp) treated as stale
 * Traces to: FGR-REQ-002 */
static void test_clock_anomaly_stale(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 5000;
    input.current_time_ms     = 3000;  /* current < timestamp */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set on clock anomaly");
    FG_TEST_PASS();
}

/* TC-UNIT-015: Disagreement detected when diff > tolerance
 * Traces to: FGR-REQ-006 */
static void test_disagreement_detected(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.redundant_airspeed_available = true;
    input.airspeed_knots              = 250.0;
    input.redundant_airspeed_knots    = 270.0;  /* diff = 20 > 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT should be set for diff > 15");
    FG_TEST_PASS();
}

/* TC-UNIT-016: No disagreement when diff <= tolerance
 * Traces to: FGR-REQ-006 */
static void test_disagreement_within_tolerance(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.redundant_airspeed_available = true;
    input.airspeed_knots              = 250.0;
    input.redundant_airspeed_knots    = 260.0;  /* diff = 10 <= 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                         "SENSOR_DISAGREEMENT should NOT be set for diff <= 15");
    FG_TEST_PASS();
}

/* TC-UNIT-017: Boundary - disagreement exactly at tolerance is NOT triggered
 * Traces to: FGR-REQ-006 */
static void test_disagreement_boundary_exact(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.redundant_airspeed_available = true;
    input.airspeed_knots              = 250.0;
    input.redundant_airspeed_knots    = 265.0;  /* diff = 15.0, NOT > 15.0 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                         "SENSOR_DISAGREEMENT should NOT be set at exact tolerance");
    FG_TEST_PASS();
}

/* TC-UNIT-018: No disagreement check when redundant is unavailable
 * Traces to: FGR-REQ-006 */
static void test_no_redundant_no_disagreement(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.redundant_airspeed_available = false;
    input.redundant_airspeed_knots     = 999.0;  /* should be ignored */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                         "SENSOR_DISAGREEMENT should NOT be set without redundant");
    FG_TEST_PASS();
}

/* TC-UNIT-019: Overspeed detected above threshold
 * Traces to: FGR-REQ-004 */
static void test_overspeed_detected(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 350.0;  /* > 340 */
    input.redundant_airspeed_knots = 351.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_OVERSPEED,
                       "OVERSPEED should be set for airspeed > 340");
    FG_TEST_PASS();
}

/* TC-UNIT-020: Boundary - airspeed at threshold is NOT overspeed
 * Traces to: FGR-REQ-004 */
static void test_overspeed_boundary_at(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 340.0;  /* NOT > 340 */
    input.redundant_airspeed_knots = 341.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_OVERSPEED,
                         "OVERSPEED should NOT be set at exact threshold");
    FG_TEST_PASS();
}

/* TC-UNIT-021: Stall risk detected above threshold
 * Traces to: FGR-REQ-005 */
static void test_stall_risk_detected(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 16.0;  /* > 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_STALL_RISK,
                       "STALL_RISK should be set for AoA > 15");
    FG_TEST_PASS();
}

/* TC-UNIT-022: Boundary - AoA at threshold is NOT stall risk
 * Traces to: FGR-REQ-005 */
static void test_stall_risk_boundary_at(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 15.0;  /* NOT > 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_STALL_RISK,
                         "STALL_RISK should NOT be set at exact threshold");
    FG_TEST_PASS();
}

/* TC-UNIT-023: Sensor invalid causes FAILSAFE
 * Traces to: FGR-REQ-003 */
static void test_sensor_invalid_failsafe(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_valid = false;
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "Mode should be FAILSAFE when sensor_valid is false");
    FG_ASSERT_EQ_INT(result.warnings, FG_WARN_NONE,
                     "No warnings should be evaluated when sensor is invalid");
    FG_TEST_PASS();
}

/* TC-UNIT-024: One sensor health warning causes DEGRADED
 * Traces to: FGR-REQ-007 */
static void test_one_health_warning_degraded(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 4000;  /* stale data */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_DEGRADED,
                     "Mode should be DEGRADED with one health warning");
    FG_TEST_PASS();
}

/* TC-UNIT-025: Two sensor health warnings cause FAILSAFE
 * Traces to: FGR-REQ-008 */
static void test_two_health_warnings_failsafe(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    /* Stale data + sensor disagreement */
    input.sensor_timestamp_ms      = 1000;
    input.current_time_ms          = 4000;  /* stale */
    input.redundant_airspeed_knots = 280.0; /* disagreement: |250-280| = 30 > 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "Mode should be FAILSAFE with two health warnings");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_DATA_STALE,
                       "DATA_STALE should be set");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT should be set");
    FG_TEST_PASS();
}

/* TC-UNIT-026: Overspeed alone does NOT degrade mode
 * Traces to: FGR-REQ-014 */
static void test_overspeed_alone_normal_mode(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 350.0;
    input.redundant_airspeed_knots = 351.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Mode should stay NORMAL with only OVERSPEED");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_OVERSPEED,
                       "OVERSPEED flag should be set");
    FG_TEST_PASS();
}

/* TC-UNIT-027: Stall risk alone does NOT degrade mode
 * Traces to: FGR-REQ-014 */
static void test_stall_risk_alone_normal_mode(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.angle_of_attack_deg = 18.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_NORMAL,
                     "Mode should stay NORMAL with only STALL_RISK");
    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_STALL_RISK,
                       "STALL_RISK flag should be set");
    FG_TEST_PASS();
}

/* TC-UNIT-028: Invalid range skips overspeed/stall checks
 * Traces to: FGR-REQ-001, FGR-REQ-004, FGR-REQ-005 */
static void test_invalid_range_skips_flight_checks(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    /* Airspeed out of range AND would be overspeed if checked */
    input.airspeed_knots           = 600.0;  /* > 500 (invalid), > 340 (overspeed) */
    input.redundant_airspeed_knots = 601.0;
    input.angle_of_attack_deg      = 25.0;   /* > 15 (stall risk) */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_INVALID_RANGE,
                       "INVALID_RANGE should be set");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_OVERSPEED,
                         "OVERSPEED should NOT be set when range is invalid");
    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_STALL_RISK,
                         "STALL_RISK should NOT be set when range is invalid");
    FG_TEST_PASS();
}

/* TC-UNIT-029: Mode string conversion
 * Traces to: utility function */
static void test_mode_to_string(void)
{
    FG_ASSERT(strcmp(fg_mode_to_string(FG_MODE_NORMAL),   "NORMAL")   == 0,
              "NORMAL string mismatch");
    FG_ASSERT(strcmp(fg_mode_to_string(FG_MODE_DEGRADED), "DEGRADED") == 0,
              "DEGRADED string mismatch");
    FG_ASSERT(strcmp(fg_mode_to_string(FG_MODE_FAILSAFE), "FAILSAFE") == 0,
              "FAILSAFE string mismatch");
    FG_ASSERT(strcmp(fg_mode_to_string((FG_SystemMode)99), "UNKNOWN") == 0,
              "Invalid mode should return UNKNOWN");
    FG_TEST_PASS();
}

/* TC-UNIT-030: Warning string conversion - no warnings
 * Traces to: utility function */
static void test_warnings_to_string_none(void)
{
    char buf[256];
    int written = fg_warnings_to_string(FG_WARN_NONE, buf, sizeof(buf));

    FG_ASSERT(written > 0, "Should write characters");
    FG_ASSERT(strcmp(buf, "NONE") == 0, "Empty warnings should produce NONE");
    FG_TEST_PASS();
}

/* TC-UNIT-031: Warning string conversion - multiple flags
 * Traces to: utility function */
static void test_warnings_to_string_multiple(void)
{
    char buf[256];
    uint32_t flags = (uint32_t)FG_WARN_OVERSPEED | (uint32_t)FG_WARN_DATA_STALE;
    int written = fg_warnings_to_string(flags, buf, sizeof(buf));

    FG_ASSERT(written > 0, "Should write characters");
    FG_ASSERT(strstr(buf, "OVERSPEED_WARNING") != NULL,
              "Should contain OVERSPEED_WARNING");
    FG_ASSERT(strstr(buf, "DATA_STALE") != NULL,
              "Should contain DATA_STALE");
    FG_TEST_PASS();
}

/* TC-UNIT-032: Warning string conversion - NULL buffer
 * Traces to: defensive handling */
static void test_warnings_to_string_null(void)
{
    int written = fg_warnings_to_string(FG_WARN_NONE, NULL, 256);

    FG_ASSERT_EQ_INT(written, -1, "NULL buffer should return -1");
    FG_TEST_PASS();
}

/* TC-UNIT-033: Warning string conversion - zero buffer size
 * Traces to: defensive handling */
static void test_warnings_to_string_zero_size(void)
{
    char buf[1];
    int written = fg_warnings_to_string(FG_WARN_NONE, buf, 0);

    FG_ASSERT_EQ_INT(written, -1, "Zero buffer size should return -1");
    FG_TEST_PASS();
}

/* TC-UNIT-034: Boundary - airspeed at minimum (0.0) is valid
 * Traces to: FGR-REQ-001 */
static void test_airspeed_boundary_min_valid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 0.0;
    input.redundant_airspeed_knots = 1.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_INVALID_RANGE,
                         "Airspeed 0.0 should be in valid range");
    FG_TEST_PASS();
}

/* TC-UNIT-035: Boundary - airspeed at maximum (500.0) is valid
 * Traces to: FGR-REQ-001 */
static void test_airspeed_boundary_max_valid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 500.0;
    input.redundant_airspeed_knots = 499.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_INVALID_RANGE,
                         "Airspeed 500.0 should be in valid range");
    FG_TEST_PASS();
}

/* TC-UNIT-036: Boundary - altitude at minimum (-1000.0) is valid
 * Traces to: FGR-REQ-010 */
static void test_altitude_boundary_min_valid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = -1000.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_INVALID_RANGE,
                         "Altitude -1000.0 should be in valid range");
    FG_TEST_PASS();
}

/* TC-UNIT-037: Boundary - altitude at maximum (60000.0) is valid
 * Traces to: FGR-REQ-010 */
static void test_altitude_boundary_max_valid(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.altitude_ft = 60000.0;
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_CLEAR(result.warnings, FG_WARN_INVALID_RANGE,
                         "Altitude 60000.0 should be in valid range");
    FG_TEST_PASS();
}

/* TC-UNIT-038: Three sensor health warnings cause FAILSAFE
 * Traces to: FGR-REQ-008 */
static void test_three_health_warnings_failsafe(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    /* Invalid range + stale data + disagreement */
    input.airspeed_knots           = 600.0;   /* out of range */
    input.sensor_timestamp_ms      = 1000;
    input.current_time_ms          = 4000;    /* stale */
    input.redundant_airspeed_knots = 800.0;   /* disagreement */
    fg_evaluate(&input, &result);

    FG_ASSERT_EQ_INT(result.mode, FG_MODE_FAILSAFE,
                     "Mode should be FAILSAFE with three health warnings");
    FG_TEST_PASS();
}

/* TC-UNIT-039: Negative disagreement (redundant > primary) also detected
 * Traces to: FGR-REQ-006 */
static void test_disagreement_negative_diff(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result;

    input.airspeed_knots           = 200.0;
    input.redundant_airspeed_knots = 220.0;  /* diff = -20, |diff| = 20 > 15 */
    fg_evaluate(&input, &result);

    FG_ASSERT_FLAG_SET(result.warnings, FG_WARN_SENSOR_DISAGREEMENT,
                       "SENSOR_DISAGREEMENT should detect negative diff");
    FG_TEST_PASS();
}

/* TC-UNIT-040: Determinism - same input always produces same output
 * Traces to: FGR-REQ-013 */
static void test_determinism(void)
{
    FG_SensorInput input = make_nominal();
    FG_EvalResult result1, result2;

    fg_evaluate(&input, &result1);
    fg_evaluate(&input, &result2);

    FG_ASSERT_EQ_INT(result1.mode, result2.mode,
                     "Mode should be identical for repeated evaluations");
    FG_ASSERT_EQ_INT((int)result1.warnings, (int)result2.warnings,
                     "Warnings should be identical for repeated evaluations");
    FG_TEST_PASS();
}

/* ========================================================================
 * Test Suite Runner
 * ======================================================================== */

void run_unit_tests(void)
{
    printf("\n--- Unit Tests ---\n");

    FG_RUN_TEST(test_nominal_all_valid);
    FG_RUN_TEST(test_null_input);
    FG_RUN_TEST(test_null_result);
    FG_RUN_TEST(test_airspeed_below_range);
    FG_RUN_TEST(test_airspeed_above_range);
    FG_RUN_TEST(test_altitude_below_range);
    FG_RUN_TEST(test_altitude_above_range);
    FG_RUN_TEST(test_aoa_below_range);
    FG_RUN_TEST(test_aoa_above_range);
    FG_RUN_TEST(test_data_fresh);
    FG_RUN_TEST(test_data_stale);
    FG_RUN_TEST(test_data_stale_boundary_not_stale);
    FG_RUN_TEST(test_data_stale_boundary_stale);
    FG_RUN_TEST(test_clock_anomaly_stale);
    FG_RUN_TEST(test_disagreement_detected);
    FG_RUN_TEST(test_disagreement_within_tolerance);
    FG_RUN_TEST(test_disagreement_boundary_exact);
    FG_RUN_TEST(test_no_redundant_no_disagreement);
    FG_RUN_TEST(test_overspeed_detected);
    FG_RUN_TEST(test_overspeed_boundary_at);
    FG_RUN_TEST(test_stall_risk_detected);
    FG_RUN_TEST(test_stall_risk_boundary_at);
    FG_RUN_TEST(test_sensor_invalid_failsafe);
    FG_RUN_TEST(test_one_health_warning_degraded);
    FG_RUN_TEST(test_two_health_warnings_failsafe);
    FG_RUN_TEST(test_overspeed_alone_normal_mode);
    FG_RUN_TEST(test_stall_risk_alone_normal_mode);
    FG_RUN_TEST(test_invalid_range_skips_flight_checks);
    FG_RUN_TEST(test_mode_to_string);
    FG_RUN_TEST(test_warnings_to_string_none);
    FG_RUN_TEST(test_warnings_to_string_multiple);
    FG_RUN_TEST(test_warnings_to_string_null);
    FG_RUN_TEST(test_warnings_to_string_zero_size);
    FG_RUN_TEST(test_airspeed_boundary_min_valid);
    FG_RUN_TEST(test_airspeed_boundary_max_valid);
    FG_RUN_TEST(test_altitude_boundary_min_valid);
    FG_RUN_TEST(test_altitude_boundary_max_valid);
    FG_RUN_TEST(test_three_health_warnings_failsafe);
    FG_RUN_TEST(test_disagreement_negative_diff);
    FG_RUN_TEST(test_determinism);
}
