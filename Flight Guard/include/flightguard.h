/**
 * @file flightguard.h
 * @brief FlightGuard Avionics Sensor Health Monitor - Public Interface
 *
 * This module evaluates simulated aircraft sensor data and determines
 * system operating mode and active warning flags. Designed with
 * safety-critical coding practices: no dynamic memory allocation,
 * deterministic behavior, and explicit data structures.
 *
 * DISCLAIMER: This is a portfolio project inspired by avionics verification
 * practices. It does NOT claim formal DO-178C certification or compliance.
 *
 * @version 1.0.0
 */

#ifndef FLIGHTGUARD_H
#define FLIGHTGUARD_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Version Information
 * ======================================================================== */

#define FG_VERSION_MAJOR  1
#define FG_VERSION_MINOR  0
#define FG_VERSION_PATCH  0

/* ========================================================================
 * Configuration Constants (Engineering Assumptions)
 *
 * These thresholds define the operational envelope and fault detection
 * parameters. Values are generic and intended for demonstration purposes.
 * In a real system, these would be derived from aircraft-specific data.
 * ======================================================================== */

/** Minimum valid airspeed in knots (inclusive) */
#define FG_AIRSPEED_MIN_KNOTS       0.0

/** Maximum valid airspeed in knots (inclusive) */
#define FG_AIRSPEED_MAX_KNOTS       500.0

/** Minimum valid altitude in feet (inclusive, below sea level is valid) */
#define FG_ALTITUDE_MIN_FT          (-1000.0)

/** Maximum valid altitude in feet (inclusive) */
#define FG_ALTITUDE_MAX_FT          60000.0

/** Minimum valid angle of attack in degrees (inclusive) */
#define FG_AOA_MIN_DEG              (-10.0)

/** Maximum valid angle of attack in degrees (inclusive) */
#define FG_AOA_MAX_DEG              40.0

/** Maximum age of sensor data before it is considered stale (milliseconds) */
#define FG_STALE_THRESHOLD_MS       2000U

/** Maximum allowed difference between primary and redundant airspeed (knots) */
#define FG_DISAGREE_TOLERANCE_KT    15.0

/** Airspeed above which overspeed warning is raised (knots) */
#define FG_OVERSPEED_THRESHOLD_KT   340.0

/** Angle of attack above which stall risk warning is raised (degrees) */
#define FG_STALL_AOA_THRESHOLD_DEG  15.0

/** Number of sensor health warnings that triggers FAILSAFE mode */
#define FG_FAILSAFE_WARNING_COUNT   2

/* ========================================================================
 * Enumerations
 * ======================================================================== */

/**
 * System operating mode determined by sensor health evaluation.
 *
 * NORMAL:   All sensor data is healthy; no sensor-related faults detected.
 * DEGRADED: One sensor health fault detected; system operates with caution.
 * FAILSAFE: Sensor validity lost or multiple health faults; safe fallback.
 */
typedef enum {
    FG_MODE_NORMAL   = 0,
    FG_MODE_DEGRADED = 1,
    FG_MODE_FAILSAFE = 2
} FG_SystemMode;

/**
 * Warning flags indicating specific detected conditions.
 * These are used as a bitmask in FG_EvalResult.warnings.
 *
 * Sensor health warnings (affect mode): SENSOR_DISAGREEMENT, DATA_STALE, INVALID_RANGE
 * Flight condition warnings (informational): OVERSPEED, STALL_RISK
 */
typedef enum {
    FG_WARN_NONE                = 0x00,
    FG_WARN_OVERSPEED           = 0x01,
    FG_WARN_STALL_RISK          = 0x02,
    FG_WARN_SENSOR_DISAGREEMENT = 0x04,
    FG_WARN_DATA_STALE          = 0x08,
    FG_WARN_INVALID_RANGE       = 0x10
} FG_WarningFlag;

/**
 * Bitmask of sensor health warning flags.
 * Only these warnings contribute to mode degradation/failsafe decisions.
 */
#define FG_SENSOR_HEALTH_MASK \
    (FG_WARN_SENSOR_DISAGREEMENT | FG_WARN_DATA_STALE | FG_WARN_INVALID_RANGE)

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/**
 * Sensor input data provided to the evaluation function.
 *
 * All fields must be populated by the caller. The module does not
 * retain any state between evaluations.
 */
typedef struct {
    double   airspeed_knots;            /**< Primary airspeed reading (knots) */
    double   altitude_ft;               /**< Altitude reading (feet)          */
    double   angle_of_attack_deg;       /**< Angle of attack (degrees)        */
    bool     sensor_valid;              /**< Master sensor validity flag       */
    uint64_t sensor_timestamp_ms;       /**< Timestamp of sensor data (ms)    */
    uint64_t current_time_ms;           /**< Current system time (ms)         */
    bool     redundant_airspeed_available; /**< Is redundant airspeed present? */
    double   redundant_airspeed_knots;  /**< Redundant airspeed reading (knots) */
} FG_SensorInput;

/**
 * Evaluation result produced by fg_evaluate().
 *
 * Contains the determined system mode and a bitmask of active warnings.
 */
typedef struct {
    FG_SystemMode mode;       /**< Determined system operating mode    */
    uint32_t      warnings;   /**< Bitmask of active FG_WarningFlag(s) */
} FG_EvalResult;

/* ========================================================================
 * Return Codes
 * ======================================================================== */

/** Evaluation completed successfully */
#define FG_OK          0

/** Invalid argument (NULL pointer) */
#define FG_ERR_NULL   (-1)

/* ========================================================================
 * Public API
 * ======================================================================== */

/**
 * Evaluate sensor input data and determine system mode and warnings.
 *
 * This is the primary entry point for the FlightGuard module. The function
 * is deterministic: identical inputs always produce identical outputs.
 * No dynamic memory is allocated. No global state is modified.
 *
 * @param[in]  input   Pointer to sensor input data (must not be NULL)
 * @param[out] result  Pointer to evaluation result (must not be NULL)
 * @return FG_OK on success, FG_ERR_NULL if input or result is NULL
 */
int fg_evaluate(const FG_SensorInput *input, FG_EvalResult *result);

/**
 * Convert a system mode enum value to a human-readable string.
 *
 * @param[in] mode  System mode value
 * @return Pointer to a static string literal (never NULL)
 */
const char *fg_mode_to_string(FG_SystemMode mode);

/**
 * Format warning flags bitmask into a human-readable string.
 *
 * Writes a pipe-separated list of active warning names into the
 * provided buffer. If no warnings are active, writes "NONE".
 *
 * @param[in]  warnings  Bitmask of FG_WarningFlag values
 * @param[out] buf       Output buffer (must not be NULL)
 * @param[in]  buf_size  Size of the output buffer in bytes
 * @return Number of characters written (excluding null terminator),
 *         or -1 if buf is NULL or buf_size is 0
 */
int fg_warnings_to_string(uint32_t warnings, char *buf, size_t buf_size);

#endif /* FLIGHTGUARD_H */
