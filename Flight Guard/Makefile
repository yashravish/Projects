# =============================================================================
# FlightGuard: Avionics Sensor Health Monitor - Makefile
#
# Targets:
#   all       - Build the demo executable (default)
#   test      - Build and run the full test suite
#   sim       - Build and run the HIL-style simulator
#   run       - Build and run the demo executable
#   clean     - Remove all build artifacts
#   coverage  - Build with gcov instrumentation and generate coverage data
# =============================================================================

CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude
LDFLAGS  =

# Directories
SRC_DIR   = src
INC_DIR   = include
TEST_DIR  = test
SIM_DIR   = sim
BUILD_DIR = build

# Source files
CORE_SRC  = $(SRC_DIR)/flightguard.c
MAIN_SRC  = $(SRC_DIR)/main.c
SIM_SRC   = $(SIM_DIR)/simulator.c
TEST_SRCS = $(TEST_DIR)/test_main.c \
            $(TEST_DIR)/test_unit.c \
            $(TEST_DIR)/test_integration.c \
            $(TEST_DIR)/test_fault_injection.c

# Output executables
DEMO      = $(BUILD_DIR)/flightguard_demo
TESTS     = $(BUILD_DIR)/flightguard_tests
SIMULATOR = $(BUILD_DIR)/flightguard_sim

# =============================================================================
# Targets
# =============================================================================

.PHONY: all run test sim clean coverage

all: $(DEMO)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --- Demo executable ---
$(DEMO): $(CORE_SRC) $(MAIN_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(CORE_SRC) $(MAIN_SRC) $(LDFLAGS)

# --- Test executable ---
$(TESTS): $(CORE_SRC) $(TEST_SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TEST_DIR) -o $@ $(CORE_SRC) $(TEST_SRCS) $(LDFLAGS)

# --- Simulator executable ---
$(SIMULATOR): $(CORE_SRC) $(SIM_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(CORE_SRC) $(SIM_SRC) $(LDFLAGS)

# --- Run targets ---
run: $(DEMO)
	./$(DEMO)

test: $(TESTS)
	./$(TESTS)

sim: $(SIMULATOR)
	./$(SIMULATOR)

# --- Clean ---
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.gcno *.gcda *.gcov
	rm -f $(SRC_DIR)/*.gcno $(SRC_DIR)/*.gcda
	rm -f $(TEST_DIR)/*.gcno $(TEST_DIR)/*.gcda

# --- Coverage (gcov) ---
coverage: $(CORE_SRC) $(TEST_SRCS)
	mkdir -p $(BUILD_DIR)/coverage
	$(CC) $(CFLAGS) -I$(TEST_DIR) --coverage -o $(BUILD_DIR)/flightguard_tests_cov \
		$(CORE_SRC) $(TEST_SRCS) $(LDFLAGS)
	./$(BUILD_DIR)/flightguard_tests_cov
	gcov $(CORE_SRC)
	mv -f *.gcov $(BUILD_DIR)/coverage/ 2>/dev/null || true
	mv -f *.gcno *.gcda $(BUILD_DIR)/coverage/ 2>/dev/null || true
	@echo ""
	@echo "=== Coverage data generated in $(BUILD_DIR)/coverage/ ==="
	@echo "Inspect .gcov files for line-by-line coverage annotation."
