# FlightGuard Code Review Checklist

**Document ID:** FG-CRC-001  
**Version:** 1.0  
**Project:** FlightGuard Avionics Sensor Health Monitor  

---

## Purpose

This checklist guides code reviews for the FlightGuard project. It is
designed to catch common issues in safety-critical embedded software
and ensure consistency with project standards.

---

## 1. Logic Correctness

- [ ] All comparison operators are correct (e.g., `>` vs `>=`, `<` vs `<=`)
- [ ] Boundary conditions are handled correctly at thresholds
- [ ] Boolean logic has no unintended short-circuit effects
- [ ] Switch statements have default cases
- [ ] No integer overflow or underflow in arithmetic operations
- [ ] Floating-point comparisons use appropriate precision
- [ ] No division by zero possible
- [ ] All code paths return a value or produce defined output

## 2. Boundary Handling

- [ ] Minimum and maximum valid values are accepted correctly
- [ ] Values just outside valid range are rejected correctly
- [ ] Edge cases for timestamp values (zero, rollback) are handled
- [ ] Disagreement tolerance boundary (exact vs. just over) is correct
- [ ] Overspeed and stall risk thresholds use strict comparison (`>`)
- [ ] Stale data threshold uses strict comparison (`>`)

## 3. Traceability

- [ ] Each function has a comment tracing to requirement ID(s)
- [ ] Each test case has a comment tracing to requirement ID(s)
- [ ] traceability_matrix.md is up to date
- [ ] New requirements have corresponding tests
- [ ] No orphan tests (tests without requirement links)

## 4. Naming Conventions

- [ ] Functions use `fg_` prefix for public API
- [ ] Internal functions use `fg_` prefix and are declared `static`
- [ ] Constants use `FG_` prefix with UPPER_SNAKE_CASE
- [ ] Struct types use `FG_` prefix with PascalCase
- [ ] Enum values use `FG_` prefix with UPPER_SNAKE_CASE
- [ ] Variable names are descriptive and unambiguous
- [ ] No single-letter variable names (except loop indices)

## 5. Comments and Documentation

- [ ] File-level header comments describe purpose and scope
- [ ] Each public function has a documentation comment (Doxygen-style)
- [ ] Complex logic has inline explanatory comments
- [ ] Comments explain "why" rather than "what" where appropriate
- [ ] No commented-out code remains
- [ ] No TODO or FIXME markers remain unresolved
- [ ] Engineering assumptions are documented near their usage

## 6. Determinism

- [ ] No calls to `rand()`, `time()`, or other non-deterministic functions
- [ ] No dependency on execution order beyond explicit control flow
- [ ] No use of uninitialized variables
- [ ] Output depends only on input parameters (no hidden state)
- [ ] No floating-point mode-dependent behavior

## 7. Test Adequacy

- [ ] Every requirement has at least one test
- [ ] Boundary values are explicitly tested for all thresholds
- [ ] Both positive and negative cases are tested
- [ ] Error paths (NULL pointers, invalid inputs) are tested
- [ ] Combined fault scenarios are covered
- [ ] Tests verify both the presence and absence of warning flags
- [ ] Tests verify mode and warnings together (not just individually)
- [ ] Tests are independent (no shared state between tests)

## 8. Error Handling

- [ ] NULL pointer arguments are checked before dereferencing
- [ ] Function return codes are defined and documented
- [ ] Error conditions do not cause undefined behavior
- [ ] Error conditions produce predictable, safe output
- [ ] Buffer size limits are respected in string operations

## 9. Memory and Resources

- [ ] No dynamic memory allocation (`malloc`, `calloc`, `realloc`, `free`)
- [ ] No global mutable state
- [ ] Stack usage is bounded (no deep recursion)
- [ ] No memory leaks possible
- [ ] All buffers have explicit size limits

## 10. Maintainability

- [ ] Functions are small and focused (single responsibility)
- [ ] No duplicate logic between functions
- [ ] Configuration values are #defined constants, not magic numbers
- [ ] Enum values are used instead of integer literals
- [ ] File organization follows a logical structure
- [ ] Build system is clean and documented
- [ ] Code compiles without warnings under strict flags

## 11. Safety-Critical Practices

- [ ] No use of `goto`
- [ ] No function pointers (unless justified and documented)
- [ ] No recursion
- [ ] No unions (unless justified and documented)
- [ ] Casts are explicit and justified
- [ ] Integer types have explicit width (`uint32_t`, not `unsigned`)
- [ ] No reliance on implementation-defined behavior
- [ ] Side effects are limited to output parameters

---

## Review Summary Template

| Item | Status | Notes |
|------|--------|-------|
| Reviewer | | |
| Date | | |
| Files Reviewed | | |
| Logic Correct | Pass / Fail | |
| Boundaries Handled | Pass / Fail | |
| Traceability Current | Pass / Fail | |
| Naming Consistent | Pass / Fail | |
| Documentation Complete | Pass / Fail | |
| Tests Adequate | Pass / Fail | |
| Overall Verdict | Accept / Rework | |
