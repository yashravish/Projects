# CUDA Audio Signal Processor MVP

A hands-on introduction to CUDA programming through audio signal processing - relevant for music tech applications.

## Project Structure

```
.
|-- vector_add.cu       # The "Hello World" of GPU computing
|-- audio_gain.cu       # Volume control with GPU
|-- audio_mixer.cu      # Mix multiple audio signals
|-- simple_filter.cu    # Low-pass filter concept
|-- Makefile            # Build all examples
|-- README.md
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc --version` to verify)
- C/C++ compiler (gcc/g++)

## Quick Start

```bash
# Build all examples
make all

# Run examples
./vector_add
./audio_gain
./audio_mixer
./simple_filter
```

## Build and Validation

See `BUILDING.md` for detailed, copy-paste build and validation steps.

## Learning Path

### Stage 1: CUDA Fundamentals (`vector_add.cu`)
- Host vs Device memory
- Kernel functions with `__global__`
- Thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- Memory transfers: `cudaMalloc`, `cudaMemcpy`, `cudaFree`

### Stage 2: Audio Processing (`audio_gain.cu`, `audio_mixer.cu`)
- Apply CUDA to real audio data (float samples)
- Parallel gain adjustment (volume control)
- Signal mixing (combining multiple audio sources)

### Stage 3: Frequency Domain (`simple_filter.cu`)
- Simple time-domain filtering
- Foundation for FFT-based processing (future expansion)

## Key CUDA Concepts Demonstrated

| Concept | File | Description |
|---------|------|-------------|
| Kernel Launch | `vector_add.cu` | `<<<blocks, threads>>>` syntax |
| Global Memory | All files | `cudaMalloc`, `cudaMemcpy` |
| Thread Indexing | All files | Unique thread ID calculation |
| Error Handling | All files | `cudaGetLastError()` checking |
| Grid/Block Design | All files | Optimal thread organization |

## For Your Music Tech Portfolio

This MVP demonstrates:
1. **GPU parallel processing** - essential for real-time audio
2. **Memory management** - critical for low-latency applications
3. **Scalable architecture** - handles large audio buffers efficiently

Future extensions for Carnatic music applications:
- Real-time pitch detection using GPU FFT
- Parallel spectrogram generation
- GPU-accelerated audio feature extraction for ML models
