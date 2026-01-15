# CUDA Audio Signal Processor MVP - Makefile
# 
# Usage:
#   make all      - Build all examples
#   make clean    - Remove all binaries
#   make run      - Build and run all examples
#   make <name>   - Build specific example (e.g., make vector_add)

# CUDA compiler
# Override with: make NVCC=/path/to/nvcc
NVCC ?= nvcc

# Compiler flags
NVCC_FLAGS = -O2 -std=c++17 -arch=sm_50
STRICT_FLAGS = $(NVCC_FLAGS) -Wall -Wextra -Werror

# Detect GPU architecture (optional, uncomment to auto-detect)
# GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
# NVCC_FLAGS = -O2 -arch=sm_$(GPU_ARCH)

# Targets
TARGETS = vector_add \
          audio_gain \
          audio_mixer \
          simple_filter

# Default target
all: $(TARGETS)
	@echo ""
	@echo "All examples built successfully!"
	@echo ""
	@echo "Run individual examples:"
	@echo "  ./vector_add"
	@echo "  ./audio_gain"
	@echo "  ./audio_mixer"
	@echo "  ./simple_filter"

# Build rules
vector_add: vector_add.cu
	@echo "Building vector_add..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

audio_gain: audio_gain.cu
	@echo "Building audio_gain..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

audio_mixer: audio_mixer.cu
	@echo "Building audio_mixer..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

simple_filter: simple_filter.cu
	@echo "Building simple_filter..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Strict build (warnings as errors)
strict: NVCC_FLAGS := $(STRICT_FLAGS)
strict: all

# Run all examples
run: all
	@echo ""
	@echo "========================================="
	@echo "Running: Vector Addition"
	@echo "========================================="
	./vector_add
	@echo ""
	@echo "========================================="
	@echo "Running: Audio Gain"
	@echo "========================================="
	./audio_gain
	@echo ""
	@echo "========================================="
	@echo "Running: Audio Mixer"
	@echo "========================================="
	./audio_mixer
	@echo ""
	@echo "========================================="
	@echo "Running: Simple Filter"
	@echo "========================================="
	./simple_filter

# Clean build artifacts
clean:
	rm -f $(TARGETS)
	@echo "Cleaned all binaries."

# Help
help:
	@echo "CUDA Audio MVP - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples"
	@echo "  clean        - Remove all binaries"
	@echo "  run          - Build and run all examples"
	@echo "  vector_add   - Build just vector addition"
	@echo "  audio_gain   - Build just audio gain"
	@echo "  audio_mixer  - Build just audio mixer"
	@echo "  simple_filter- Build just simple filter"
	@echo "  strict       - Build all with -Wall -Wextra -Werror"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - NVIDIA GPU with CUDA support"
	@echo "  - CUDA Toolkit (nvcc compiler)"

.PHONY: all clean run help strict vector_add audio_gain audio_mixer simple_filter
