/**
 * @file test_main.c
 * @brief FlightGuard Test Runner
 *
 * Entry point for the FlightGuard test suite. Executes all test suites
 * (unit, integration, fault injection) and reports aggregate results.
 */

#include <stdio.h>

/* Global test counters used by test_harness.h macros */
int fg_tests_run    = 0;
int fg_tests_passed = 0;
int fg_tests_failed = 0;

/* Test suite declarations (defined in respective source files) */
extern void run_unit_tests(void);
extern void run_integration_tests(void);
extern void run_fault_injection_tests(void);

int main(void)
{
    printf("============================================================\n");
    printf("  FlightGuard Test Suite\n");
    printf("============================================================\n");

    run_unit_tests();
    run_integration_tests();
    run_fault_injection_tests();

    /* Print summary */
    printf("\n========================================\n");
    printf("  Test Results Summary\n");
    printf("========================================\n");
    printf("  Total:  %d\n", fg_tests_run);
    printf("  Passed: %d\n", fg_tests_passed);
    printf("  Failed: %d\n", fg_tests_failed);
    printf("========================================\n");

    if (fg_tests_failed == 0) {
        printf("  STATUS: ALL TESTS PASSED\n");
    } else {
        printf("  STATUS: %d TEST(S) FAILED\n", fg_tests_failed);
    }

    printf("========================================\n");

    /* Return non-zero exit code if any tests failed (for CI) */
    return (fg_tests_failed > 0) ? 1 : 0;
}
