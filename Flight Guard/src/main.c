/**
 * @file main.c
 * @brief FlightGuard Demo - Single Evaluation Entry Point
 *
 * Demonstrates a single invocation of the FlightGuard sensor health
 * evaluation module with nominal input data. This executable serves
 * as a quick smoke test and portfolio demonstration.
 */

#include <stdio.h>
#include "flightguard.h"

/**
 * Print a formatted sensor input summary to stdout.
 */
static void print_input(const FG_SensorInput *input)
{
    printf("  Airspeed (primary):   %.1f knots\n", input->airspeed_knots);
    printf("  Altitude:             %.1f ft\n", input->altitude_ft);
    printf("  Angle of Attack:      %.1f deg\n", input->angle_of_attack_deg);
    printf("  Sensor Valid:         %s\n", input->sensor_valid ? "YES" : "NO");
    printf("  Sensor Timestamp:     %llu ms\n",
           (unsigned long long)input->sensor_timestamp_ms);
    printf("  Current Time:         %llu ms\n",
           (unsigned long long)input->current_time_ms);
    printf("  Redundant Available:  %s\n",
           input->redundant_airspeed_available ? "YES" : "NO");

    if (input->redundant_airspeed_available) {
        printf("  Airspeed (redundant): %.1f knots\n",
               input->redundant_airspeed_knots);
    }
}

/**
 * Print a formatted evaluation result summary to stdout.
 */
static void print_result(const FG_EvalResult *result)
{
    char warn_buf[256];

    fg_warnings_to_string(result->warnings, warn_buf, sizeof(warn_buf));

    printf("  System Mode: %s\n", fg_mode_to_string(result->mode));
    printf("  Warnings:    %s\n", warn_buf);
}

int main(void)
{
    printf("============================================================\n");
    printf("  FlightGuard: Avionics Sensor Health Monitor v%d.%d.%d\n",
           FG_VERSION_MAJOR, FG_VERSION_MINOR, FG_VERSION_PATCH);
    printf("============================================================\n\n");

    /* Construct a nominal sensor input representing stable cruise */
    FG_SensorInput input;
    input.airspeed_knots            = 250.0;
    input.altitude_ft               = 35000.0;
    input.angle_of_attack_deg       = 3.5;
    input.sensor_valid              = true;
    input.sensor_timestamp_ms       = 1000;
    input.current_time_ms           = 1500;
    input.redundant_airspeed_available = true;
    input.redundant_airspeed_knots  = 252.0;

    FG_EvalResult result;

    printf("--- Demo: Nominal Cruise Evaluation ---\n\n");
    printf("Input:\n");
    print_input(&input);

    int rc = fg_evaluate(&input, &result);

    if (rc != FG_OK) {
        printf("\nERROR: Evaluation failed (return code: %d)\n", rc);
        return 1;
    }

    printf("\nResult:\n");
    print_result(&result);

    /* Second demo: degraded scenario */
    printf("\n--- Demo: Degraded Scenario (Stale Data) ---\n\n");

    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 5000;  /* 4000ms age > 2000ms threshold */

    printf("Input:\n");
    print_input(&input);

    rc = fg_evaluate(&input, &result);

    if (rc != FG_OK) {
        printf("\nERROR: Evaluation failed (return code: %d)\n", rc);
        return 1;
    }

    printf("\nResult:\n");
    print_result(&result);

    /* Third demo: failsafe scenario */
    printf("\n--- Demo: FAILSAFE Scenario (Sensor Invalid) ---\n\n");

    input.sensor_valid        = false;
    input.sensor_timestamp_ms = 1000;
    input.current_time_ms     = 1500;

    printf("Input:\n");
    print_input(&input);

    rc = fg_evaluate(&input, &result);

    if (rc != FG_OK) {
        printf("\nERROR: Evaluation failed (return code: %d)\n", rc);
        return 1;
    }

    printf("\nResult:\n");
    print_result(&result);

    printf("\n============================================================\n");
    printf("  Demo complete. All evaluations executed successfully.\n");
    printf("============================================================\n");

    return 0;
}
