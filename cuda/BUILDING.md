# Build and Validation Guide

This repository is a CUDA C++ learning project. All `.cu` sources are located at the repo root.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc --version` should work)
- C/C++ compiler (MSVC, gcc, or clang)
- `make` (optional, but recommended)

## Verify CUDA Toolkit

```bash
nvcc --version
```

If `nvcc` is not found, ensure the CUDA Toolkit `bin` directory is on your `PATH`.

## Build with Makefile (preferred)

```bash
make all
make strict
```

Per-target builds:

```bash
make vector_add
make audio_gain
make audio_mixer
make simple_filter
```

Clean:

```bash
make clean
```

### Overriding the CUDA compiler path

If `nvcc` is not on `PATH`, pass `NVCC` directly:

```bash
make NVCC=/path/to/nvcc all
```

## No-make fallback (explicit commands)

```bash
nvcc -O2 -std=c++17 vector_add.cu -o vector_add
nvcc -O2 -std=c++17 audio_gain.cu -o audio_gain
nvcc -O2 -std=c++17 audio_mixer.cu -o audio_mixer
nvcc -O2 -std=c++17 simple_filter.cu -o simple_filter
```

Strict build (warnings as errors):

```bash
nvcc -O2 -std=c++17 -Wall -Wextra -Werror vector_add.cu -o vector_add
```

## Run the binaries

```bash
./vector_add
./audio_gain
./audio_mixer
./simple_filter
```

### Expected high-level success signals

- `vector_add`: prints "SUCCESS: GPU results match CPU results!"
- `audio_gain`: prints peak/RMS before and after gain tests
- `audio_mixer`: prints per-track levels and mixed output metrics
- `simple_filter`: prints ZCR/RMS changes across filter tests

## Windows notes

PowerShell:

```powershell
nvcc --version
make all
.\vector_add.exe
```

If `make` is unavailable on Windows, use the explicit `nvcc` commands above.
WSL (Ubuntu) can be the simplest route for `make` usage if CUDA is configured in WSL.

