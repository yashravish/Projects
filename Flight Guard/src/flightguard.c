/**
 * @file flightguard.c
 * @brief FlightGuard Avionics Sensor Health Monitor - Core Implementation
 *
 * Implements deterministic sensor health evaluation logic. Each check is
 * isolated in its own function for clarity, testability, and traceability
 * to requirements.
 *
 * Design principles:
 * - No dynamic memory allocation
 * - No global mutable state
 * - Deterministic: identical inputs produce identical outputs
 * - Small, single-purpose functions
 * - Defensive input validation
 */

#include "flightguard.h"
#include <stdio.h>
#include <string.h>

/* ========================================================================
 * Internal Helper Functions
 *
 * Each function maps to one or more requirements and performs a single
 * logical check. Functions are static to limit scope to this translation
 * unit.
 * ======================================================================== */

/**
 * Check whether airspeed is within the valid range.
 * Handles NaN implicitly: NaN comparisons return false in IEEE 754.
 *
 * Traces to: FGR-REQ-001
 */
static bool fg_is_airspeed_in_range(double airspeed_knots)
{
    return (airspeed_knots >= FG_AIRSPEED_MIN_KNOTS &&
            airspeed_knots <= FG_AIRSPEED_MAX_KNOTS);
}

/**
 * Check whether altitude is within the valid range.
 *
 * Traces to: FGR-REQ-010
 */
static bool fg_is_altitude_in_range(double altitude_ft)
{
    return (altitude_ft >= FG_ALTITUDE_MIN_FT &&
            altitude_ft <= FG_ALTITUDE_MAX_FT);
}

/**
 * Check whether angle of attack is within the valid range.
 *
 * Traces to: FGR-REQ-011
 */
static bool fg_is_aoa_in_range(double aoa_deg)
{
    return (aoa_deg >= FG_AOA_MIN_DEG &&
            aoa_deg <= FG_AOA_MAX_DEG);
}

/**
 * Check whether all sensor input values are within valid ranges.
 * Returns true if ALL values are in range, false otherwise.
 *
 * Traces to: FGR-REQ-001, FGR-REQ-010, FGR-REQ-011
 */
static bool fg_are_ranges_valid(const FG_SensorInput *input)
{
    if (!fg_is_airspeed_in_range(input->airspeed_knots)) {
        return false;
    }
    if (!fg_is_altitude_in_range(input->altitude_ft)) {
        return false;
    }
    if (!fg_is_aoa_in_range(input->angle_of_attack_deg)) {
        return false;
    }
    return true;
}

/**
 * Check whether sensor data has exceeded the staleness threshold.
 * If current_time_ms < sensor_timestamp_ms, this is treated as a
 * clock anomaly and the data is considered stale (defensive behavior).
 *
 * Traces to: FGR-REQ-002
 */
static bool fg_is_data_stale(const FG_SensorInput *input)
{
    /* Defensive: clock rollback or anomaly treated as stale */
    if (input->current_time_ms < input->sensor_timestamp_ms) {
        return true;
    }

    uint64_t age_ms = input->current_time_ms - input->sensor_timestamp_ms;
    return (age_ms > (uint64_t)FG_STALE_THRESHOLD_MS);
}

/**
 * Check whether primary and redundant airspeed values disagree
 * beyond the allowed tolerance. Only checked when redundant data
 * is available.
 *
 * Traces to: FGR-REQ-006
 */
static bool fg_is_disagreement_detected(const FG_SensorInput *input)
{
    if (!input->redundant_airspeed_available) {
        return false;
    }

    double diff = input->airspeed_knots - input->redundant_airspeed_knots;

    /* Manual absolute value to avoid math.h dependency */
    if (diff < 0.0) {
        diff = -diff;
    }

    return (diff > FG_DISAGREE_TOLERANCE_KT);
}

/**
 * Check whether airspeed exceeds the overspeed threshold.
 *
 * Traces to: FGR-REQ-004
 */
static bool fg_is_overspeed(double airspeed_knots)
{
    return (airspeed_knots > FG_OVERSPEED_THRESHOLD_KT);
}

/**
 * Check whether angle of attack exceeds the stall risk threshold.
 *
 * Traces to: FGR-REQ-005
 */
static bool fg_is_stall_risk(double aoa_deg)
{
    return (aoa_deg > FG_STALL_AOA_THRESHOLD_DEG);
}

/**
 * Count the number of set bits in a 32-bit integer.
 * Used to count active sensor health warnings.
 */
static int fg_popcount(uint32_t value)
{
    int count = 0;
    while (value != 0U) {
        count += (int)(value & 1U);
        value >>= 1;
    }
    return count;
}

/**
 * Determine the system operating mode based on sensor validity
 * and the set of active warning flags.
 *
 * Rules (evaluated in priority order):
 * 1. sensor_valid == false                        -> FAILSAFE  (FGR-REQ-003)
 * 2. Two or more sensor health warnings active    -> FAILSAFE  (FGR-REQ-008)
 * 3. Exactly one sensor health warning active     -> DEGRADED  (FGR-REQ-007)
 * 4. No sensor health warnings                    -> NORMAL    (FGR-REQ-009)
 *
 * Flight condition warnings (OVERSPEED, STALL_RISK) do not independently
 * cause mode transitions. (FGR-REQ-014)
 *
 * Traces to: FGR-REQ-003, FGR-REQ-007, FGR-REQ-008, FGR-REQ-009, FGR-REQ-014
 */
static FG_SystemMode fg_determine_mode(bool sensor_valid, uint32_t warnings)
{
    if (!sensor_valid) {
        return FG_MODE_FAILSAFE;
    }

    uint32_t health_flags = warnings & (uint32_t)FG_SENSOR_HEALTH_MASK;
    int health_count = fg_popcount(health_flags);

    if (health_count >= FG_FAILSAFE_WARNING_COUNT) {
        return FG_MODE_FAILSAFE;
    }

    if (health_count > 0) {
        return FG_MODE_DEGRADED;
    }

    return FG_MODE_NORMAL;
}

/* ========================================================================
 * Public API Implementation
 * ======================================================================== */

int fg_evaluate(const FG_SensorInput *input, FG_EvalResult *result)
{
    /* Defensive null checks */
    if (input == NULL || result == NULL) {
        return FG_ERR_NULL;
    }

    /* Initialize output to a known state */
    result->warnings = (uint32_t)FG_WARN_NONE;
    result->mode = FG_MODE_NORMAL;

    /* Rule 1: If sensor is marked invalid, enter FAILSAFE immediately.
     * No further checks are meaningful with invalid sensor data. */
    if (!input->sensor_valid) {
        result->mode = FG_MODE_FAILSAFE;
        return FG_OK;
    }

    /* Check sensor value ranges */
    if (!fg_are_ranges_valid(input)) {
        result->warnings |= (uint32_t)FG_WARN_INVALID_RANGE;
    }

    /* Check data staleness */
    if (fg_is_data_stale(input)) {
        result->warnings |= (uint32_t)FG_WARN_DATA_STALE;
    }

    /* Check redundant sensor disagreement */
    if (fg_is_disagreement_detected(input)) {
        result->warnings |= (uint32_t)FG_WARN_SENSOR_DISAGREEMENT;
    }

    /* Flight condition checks are only meaningful when range data is valid.
     * If INVALID_RANGE is set, the airspeed/AoA values cannot be trusted
     * for overspeed or stall determination. */
    if ((result->warnings & (uint32_t)FG_WARN_INVALID_RANGE) == 0U) {
        if (fg_is_overspeed(input->airspeed_knots)) {
            result->warnings |= (uint32_t)FG_WARN_OVERSPEED;
        }

        if (fg_is_stall_risk(input->angle_of_attack_deg)) {
            result->warnings |= (uint32_t)FG_WARN_STALL_RISK;
        }
    }

    /* Determine final system mode from sensor validity and warnings */
    result->mode = fg_determine_mode(input->sensor_valid, result->warnings);

    return FG_OK;
}

const char *fg_mode_to_string(FG_SystemMode mode)
{
    switch (mode) {
        case FG_MODE_NORMAL:   return "NORMAL";
        case FG_MODE_DEGRADED: return "DEGRADED";
        case FG_MODE_FAILSAFE: return "FAILSAFE";
        default:               return "UNKNOWN";
    }
}

int fg_warnings_to_string(uint32_t warnings, char *buf, size_t buf_size)
{
    if (buf == NULL || buf_size == 0) {
        return -1;
    }

    buf[0] = '\0';

    if (warnings == (uint32_t)FG_WARN_NONE) {
        return snprintf(buf, buf_size, "NONE");
    }

    /* Table of flag-to-name mappings for iteration */
    static const struct {
        uint32_t    flag;
        const char *name;
    } flag_table[] = {
        { (uint32_t)FG_WARN_OVERSPEED,           "OVERSPEED_WARNING"    },
        { (uint32_t)FG_WARN_STALL_RISK,          "STALL_RISK"           },
        { (uint32_t)FG_WARN_SENSOR_DISAGREEMENT, "SENSOR_DISAGREEMENT"  },
        { (uint32_t)FG_WARN_DATA_STALE,          "DATA_STALE"           },
        { (uint32_t)FG_WARN_INVALID_RANGE,       "INVALID_RANGE"        }
    };

    static const size_t flag_count = sizeof(flag_table) / sizeof(flag_table[0]);

    int written = 0;
    bool first = true;

    for (size_t i = 0; i < flag_count; i++) {
        if ((warnings & flag_table[i].flag) != 0U) {
            int n = snprintf(buf + written,
                             buf_size - (size_t)written,
                             "%s%s",
                             first ? "" : " | ",
                             flag_table[i].name);

            if (n < 0 || (size_t)(written + n) >= buf_size) {
                break;
            }

            written += n;
            first = false;
        }
    }

    return written;
}
