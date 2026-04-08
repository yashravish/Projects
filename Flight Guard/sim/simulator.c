/**
 * @file simulator.c
 * @brief FlightGuard HIL-Style Scenario Simulator
 *
 * Replays built-in flight scenarios through the FlightGuard evaluation
 * module, printing timestamped outputs and mode transitions. This is
 * not real hardware-in-the-loop, but demonstrates a lightweight
 * HIL-style verification bench for portfolio purposes.
 *
 * Three built-in scenarios:
 *   1. Nominal flight (takeoff -> cruise -> approach -> landing)
 *   2. Stale/disagreement fault during cruise
 *   3. Progressive sensor failure leading to FAILSAFE
 */

#include <stdio.h>
#include <string.h>
#include "flightguard.h"

/* ========================================================================
 * Scenario Data Structures
 * ======================================================================== */

/**
 * A single step in a simulation scenario.
 */
typedef struct {
    const char    *label;     /**< Description of this flight phase */
    FG_SensorInput input;     /**< Sensor input data for this step  */
} FG_ScenarioStep;

/**
 * A complete simulation scenario.
 */
typedef struct {
    const char          *name;        /**< Scenario name       */
    const char          *description; /**< Scenario description */
    const FG_ScenarioStep *steps;     /**< Array of steps      */
    int                  num_steps;   /**< Number of steps     */
} FG_Scenario;

/* ========================================================================
 * Scenario 1: Nominal Flight
 * ======================================================================== */

static const FG_ScenarioStep scenario_nominal_steps[] = {
    {
        "Takeoff Roll",
        { .airspeed_knots = 80.0,  .altitude_ft = 0.0,     .angle_of_attack_deg = 2.0,
          .sensor_valid = true, .sensor_timestamp_ms = 1000, .current_time_ms = 1100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 81.0 }
    },
    {
        "Rotation & Liftoff",
        { .airspeed_knots = 145.0, .altitude_ft = 30.0,    .angle_of_attack_deg = 10.0,
          .sensor_valid = true, .sensor_timestamp_ms = 2000, .current_time_ms = 2100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 146.0 }
    },
    {
        "Initial Climb (1000 ft)",
        { .airspeed_knots = 180.0, .altitude_ft = 1000.0,  .angle_of_attack_deg = 7.0,
          .sensor_valid = true, .sensor_timestamp_ms = 5000, .current_time_ms = 5100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 181.0 }
    },
    {
        "Climb (10000 ft)",
        { .airspeed_knots = 250.0, .altitude_ft = 10000.0, .angle_of_attack_deg = 5.0,
          .sensor_valid = true, .sensor_timestamp_ms = 15000, .current_time_ms = 15100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    },
    {
        "Cruise (FL350)",
        { .airspeed_knots = 280.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 2.5,
          .sensor_valid = true, .sensor_timestamp_ms = 60000, .current_time_ms = 60100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 281.0 }
    },
    {
        "Top of Descent",
        { .airspeed_knots = 270.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 1.5,
          .sensor_valid = true, .sensor_timestamp_ms = 120000, .current_time_ms = 120100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 271.0 }
    },
    {
        "Descent (10000 ft)",
        { .airspeed_knots = 230.0, .altitude_ft = 10000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 180000, .current_time_ms = 180100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 231.0 }
    },
    {
        "Final Approach",
        { .airspeed_knots = 140.0, .altitude_ft = 2000.0,  .angle_of_attack_deg = 6.0,
          .sensor_valid = true, .sensor_timestamp_ms = 200000, .current_time_ms = 200100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 141.0 }
    },
    {
        "Landing",
        { .airspeed_knots = 130.0, .altitude_ft = 50.0,    .angle_of_attack_deg = 8.0,
          .sensor_valid = true, .sensor_timestamp_ms = 210000, .current_time_ms = 210100,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 131.0 }
    }
};

/* ========================================================================
 * Scenario 2: Stale/Disagreement Fault
 * ======================================================================== */

static const FG_ScenarioStep scenario_fault_steps[] = {
    {
        "Normal Cruise",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 10000, .current_time_ms = 10500,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    },
    {
        "Cruise (continued)",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 11000, .current_time_ms = 11500,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    },
    {
        "FAULT: Sensor data becomes stale",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 11000, .current_time_ms = 14000,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    },
    {
        "FAULT CONTINUES: Stale data persists",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 11000, .current_time_ms = 16000,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    },
    {
        "FAULT ESCALATION: Redundant sensor drifts",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 11000, .current_time_ms = 18000,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 280.0 }
    },
    {
        "Recovery: Fresh data, redundant resynced",
        { .airspeed_knots = 250.0, .altitude_ft = 35000.0, .angle_of_attack_deg = 3.0,
          .sensor_valid = true, .sensor_timestamp_ms = 19000, .current_time_ms = 19500,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 251.0 }
    }
};

/* ========================================================================
 * Scenario 3: Progressive FAILSAFE
 * ======================================================================== */

static const FG_ScenarioStep scenario_failsafe_steps[] = {
    {
        "Normal Flight",
        { .airspeed_knots = 220.0, .altitude_ft = 25000.0, .angle_of_attack_deg = 4.0,
          .sensor_valid = true, .sensor_timestamp_ms = 5000, .current_time_ms = 5500,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 221.0 }
    },
    {
        "FAULT: Airspeed sensor reads out-of-range",
        { .airspeed_knots = 600.0, .altitude_ft = 25000.0, .angle_of_attack_deg = 4.0,
          .sensor_valid = true, .sensor_timestamp_ms = 6000, .current_time_ms = 6500,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 221.0 }
    },
    {
        "FAULT CONTINUES: Data also goes stale",
        { .airspeed_knots = 600.0, .altitude_ft = 25000.0, .angle_of_attack_deg = 4.0,
          .sensor_valid = true, .sensor_timestamp_ms = 6000, .current_time_ms = 9000,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 221.0 }
    },
    {
        "CRITICAL: Sensor validity flag drops",
        { .airspeed_knots = 600.0, .altitude_ft = 25000.0, .angle_of_attack_deg = 4.0,
          .sensor_valid = false, .sensor_timestamp_ms = 6000, .current_time_ms = 10000,
          .redundant_airspeed_available = true, .redundant_airspeed_knots = 221.0 }
    },
    {
        "FAILSAFE ACTIVE: Awaiting crew action",
        { .airspeed_knots = 0.0,   .altitude_ft = 25000.0, .angle_of_attack_deg = 0.0,
          .sensor_valid = false, .sensor_timestamp_ms = 6000, .current_time_ms = 12000,
          .redundant_airspeed_available = false, .redundant_airspeed_knots = 0.0 }
    }
};

/* ========================================================================
 * Scenario Registry
 * ======================================================================== */

static const FG_Scenario scenarios[] = {
    {
        "Nominal Flight Profile",
        "Complete flight from takeoff through cruise to landing with all sensors healthy.",
        scenario_nominal_steps,
        (int)(sizeof(scenario_nominal_steps) / sizeof(scenario_nominal_steps[0]))
    },
    {
        "Stale & Disagreement Fault",
        "Normal cruise interrupted by stale data, followed by redundant sensor drift, then recovery.",
        scenario_fault_steps,
        (int)(sizeof(scenario_fault_steps) / sizeof(scenario_fault_steps[0]))
    },
    {
        "Progressive FAILSAFE",
        "Escalating sensor failures: out-of-range data, stale data, sensor invalidation.",
        scenario_failsafe_steps,
        (int)(sizeof(scenario_failsafe_steps) / sizeof(scenario_failsafe_steps[0]))
    }
};

static const int num_scenarios = (int)(sizeof(scenarios) / sizeof(scenarios[0]));

/* ========================================================================
 * Simulator Execution
 * ======================================================================== */

/**
 * Print a separator line.
 */
static void print_separator(void)
{
    printf("------------------------------------------------------------\n");
}

/**
 * Execute a single scenario step and print results.
 */
static void execute_step(int step_num, const FG_ScenarioStep *step)
{
    FG_EvalResult result;
    char warn_buf[256];

    int rc = fg_evaluate(&step->input, &result);

    if (rc != FG_OK) {
        printf("  [Step %d] ERROR: Evaluation failed (rc=%d)\n", step_num, rc);
        return;
    }

    fg_warnings_to_string(result.warnings, warn_buf, sizeof(warn_buf));

    printf("  [Step %2d] %-40s | T=%7llu ms\n",
           step_num, step->label,
           (unsigned long long)step->input.current_time_ms);
    printf("            AS=%.0f kt  ALT=%.0f ft  AoA=%.1f deg  "
           "Valid=%s  Age=%llu ms\n",
           step->input.airspeed_knots,
           step->input.altitude_ft,
           step->input.angle_of_attack_deg,
           step->input.sensor_valid ? "Y" : "N",
           step->input.sensor_valid ?
               (unsigned long long)(step->input.current_time_ms -
                                    step->input.sensor_timestamp_ms) : 0ULL);

    if (step->input.redundant_airspeed_available) {
        printf("            Redundant AS=%.0f kt\n",
               step->input.redundant_airspeed_knots);
    }

    printf("            >> Mode: %-10s  Warnings: %s\n",
           fg_mode_to_string(result.mode), warn_buf);
    printf("\n");
}

/**
 * Execute a complete scenario.
 */
static void run_scenario(const FG_Scenario *scenario)
{
    printf("\n");
    print_separator();
    printf("  SCENARIO: %s\n", scenario->name);
    printf("  %s\n", scenario->description);
    print_separator();
    printf("\n");

    for (int i = 0; i < scenario->num_steps; i++) {
        execute_step(i + 1, &scenario->steps[i]);
    }
}

/* ========================================================================
 * Main Entry Point
 * ======================================================================== */

int main(void)
{
    printf("============================================================\n");
    printf("  FlightGuard HIL-Style Scenario Simulator v%d.%d.%d\n",
           FG_VERSION_MAJOR, FG_VERSION_MINOR, FG_VERSION_PATCH);
    printf("============================================================\n");
    printf("  Replaying %d built-in scenarios...\n", num_scenarios);

    for (int i = 0; i < num_scenarios; i++) {
        run_scenario(&scenarios[i]);
    }

    printf("============================================================\n");
    printf("  Simulation complete. All scenarios replayed successfully.\n");
    printf("============================================================\n");

    return 0;
}
