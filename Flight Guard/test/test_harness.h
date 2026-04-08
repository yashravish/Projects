/**
 * @file test_harness.h
 * @brief Minimal C Test Harness for FlightGuard
 *
 * Provides assertion macros and test runner infrastructure without
 * external dependencies. Designed for deterministic, requirements-based
 * testing of safety-critical code.
 *
 * Usage:
 *   - Each test function returns void and uses FG_ASSERT macros
 *   - On assertion failure, the test increments fail count and returns
 *   - On success (reaching end of function), call FG_TEST_PASS()
 *   - Use FG_RUN_TEST() to execute and report each test
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>

/* ========================================================================
 * Global Test Counters (defined in test_main.c)
 * ======================================================================== */

extern int fg_tests_run;
extern int fg_tests_passed;
extern int fg_tests_failed;

/* ========================================================================
 * Assertion Macros
 *
 * On failure: prints location and message, increments counters, returns.
 * On success: execution continues to next assertion or FG_TEST_PASS().
 * ======================================================================== */

/**
 * Assert that a condition is true. Fails the test if condition is false.
 */
#define FG_ASSERT(cond, msg) do {                                    \
    if (!(cond)) {                                                   \
        printf("    FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
        fg_tests_failed++;                                           \
        fg_tests_run++;                                              \
        return;                                                      \
    }                                                                \
} while (0)

/**
 * Assert that two integer values are equal.
 */
#define FG_ASSERT_EQ_INT(actual, expected, msg) do {                      \
    int _fg_a = (int)(actual);                                            \
    int _fg_e = (int)(expected);                                          \
    if (_fg_a != _fg_e) {                                                 \
        printf("    FAIL [%s:%d]: %s (expected %d, got %d)\n",            \
               __FILE__, __LINE__, (msg), _fg_e, _fg_a);                  \
        fg_tests_failed++;                                                \
        fg_tests_run++;                                                   \
        return;                                                           \
    }                                                                     \
} while (0)

/**
 * Assert that a condition is true (alias for readability).
 */
#define FG_ASSERT_TRUE(cond, msg)  FG_ASSERT((cond), (msg))

/**
 * Assert that a condition is false.
 */
#define FG_ASSERT_FALSE(cond, msg) FG_ASSERT(!(cond), (msg))

/**
 * Assert that a warning flag IS set in the warnings bitmask.
 */
#define FG_ASSERT_FLAG_SET(warnings, flag, msg) \
    FG_ASSERT(((warnings) & (uint32_t)(flag)) != 0U, (msg))

/**
 * Assert that a warning flag is NOT set in the warnings bitmask.
 */
#define FG_ASSERT_FLAG_CLEAR(warnings, flag, msg) \
    FG_ASSERT(((warnings) & (uint32_t)(flag)) == 0U, (msg))

/**
 * Mark the current test as passed. Must be called at the end of
 * every test function that did not fail an assertion.
 */
#define FG_TEST_PASS() do {  \
    fg_tests_passed++;       \
    fg_tests_run++;          \
} while (0)

/* ========================================================================
 * Test Runner Macro
 * ======================================================================== */

/**
 * Run a single test function and report PASS/FAIL.
 * Tracks the failure count before execution to detect assertion failures.
 */
#define FG_RUN_TEST(fn) do {                            \
    int _fg_before = fg_tests_failed;                   \
    fn();                                               \
    if (fg_tests_failed == _fg_before) {                \
        printf("  [PASS] %s\n", #fn);                  \
    } else {                                            \
        printf("  [FAIL] %s\n", #fn);                   \
    }                                                   \
} while (0)

/* ========================================================================
 * Summary Macro
 * ======================================================================== */

/**
 * Print a summary of all test results.
 */
#define FG_TEST_SUMMARY() do {                                          \
    printf("\n========================================\n");              \
    printf("  Test Results Summary\n");                                 \
    printf("========================================\n");               \
    printf("  Total:  %d\n", fg_tests_run);                            \
    printf("  Passed: %d\n", fg_tests_passed);                         \
    printf("  Failed: %d\n", fg_tests_failed);                         \
    printf("========================================\n");               \
    if (fg_tests_failed == 0) {                                        \
        printf("  STATUS: ALL TESTS PASSED\n");                        \
    } else {                                                           \
        printf("  STATUS: %d TEST(S) FAILED\n", fg_tests_failed);     \
    }                                                                  \
    printf("========================================\n");               \
} while (0)

#endif /* TEST_HARNESS_H */
