# CUDA Quick Reference Cheatsheet

## Core Concepts

### Host vs Device
```
Host   = CPU + System RAM
Device = GPU + Video RAM (VRAM)
```

### Memory Hierarchy
```
Global Memory  → Largest, slowest (accessible by all threads)
Shared Memory  → Per-block, fast (accessible by threads in same block)
Registers      → Per-thread, fastest (local variables)
```

## Function Qualifiers

| Qualifier | Runs On | Called From |
|-----------|---------|-------------|
| `__global__` | GPU | CPU (or GPU with dynamic parallelism) |
| `__device__` | GPU | GPU only |
| `__host__` | CPU | CPU only |

## Thread Organization

```
Grid
├── Block (0,0)
│   ├── Thread (0,0,0)
│   ├── Thread (1,0,0)
│   └── ...
├── Block (1,0)
└── Block (2,0)
```

### Built-in Variables
```c
// Thread position
threadIdx.x, threadIdx.y, threadIdx.z   // Within block
blockIdx.x, blockIdx.y, blockIdx.z      // Block within grid
blockDim.x, blockDim.y, blockDim.z      // Threads per block
gridDim.x, gridDim.y, gridDim.z         // Blocks per grid
```

### Calculate Unique Thread ID (1D)
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### Calculate for 2D Grid
```c
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

## Memory Management

### Allocation
```c
// Host (CPU)
float *h_data = (float*)malloc(size);

// Device (GPU)
float *d_data;
cudaMalloc(&d_data, size);
```

### Data Transfer
```c
// Host → Device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Device → Device
cudaMemcpy(d_dest, d_src, size, cudaMemcpyDeviceToDevice);
```

### Deallocation
```c
free(h_data);
cudaFree(d_data);
```

## Kernel Launch

```c
// Syntax: kernelName<<<numBlocks, threadsPerBlock>>>(args);

// 1D Example
int threadsPerBlock = 256;
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
myKernel<<<numBlocks, threadsPerBlock>>>(d_data, N);

// 2D Example
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
myKernel2D<<<gridSize, blockSize>>>(d_data, width, height);
```

## Error Handling

```c
// Check last error
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}

// Macro for checking calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
```

## Synchronization

```c
// Wait for all GPU operations to complete
cudaDeviceSynchronize();

// Within a kernel - wait for all threads in block
__syncthreads();
```

## Common Kernel Patterns

### Map (Per-Element Operation)
```c
__global__ void map(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Example operation
    }
}
```

### Reduce (Sum, Max, etc.)
```c
__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

## Compilation

```bash
# Basic compilation
nvcc -o program program.cu

# With architecture flag (compute capability 5.0)
nvcc -arch=sm_50 -o program program.cu

# With optimization
nvcc -O2 -arch=sm_50 -o program program.cu

# Debug build
nvcc -g -G -o program program.cu
```

## Query GPU Info

```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // Device 0

printf("GPU: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Total global memory: %zu GB\n", prop.totalGlobalMem / (1024*1024*1024));
```

## Performance Tips

1. **Coalesced Memory Access**: Access consecutive memory addresses from consecutive threads
2. **Avoid Branch Divergence**: Threads in a warp should take the same path
3. **Use Shared Memory**: For data reused by multiple threads in a block
4. **Occupancy**: Launch enough threads to hide memory latency
5. **Minimize Host-Device Transfers**: Keep data on GPU as long as possible

## Audio-Specific Considerations

| Operation | Parallelization Strategy |
|-----------|-------------------------|
| Gain/Volume | Perfect: 1 thread per sample |
| Mixing | Perfect: 1 thread per sample |
| FFT | Use cuFFT library |
| IIR Filter | Sequential per-channel, parallel across channels |
| FIR Filter | Parallel: compute each output sample independently |
| Convolution | Overlap-add with FFT for efficiency |

## Useful Libraries

- **cuFFT** - Fast Fourier Transform
- **cuBLAS** - Linear algebra
- **Thrust** - STL-like algorithms for GPU
- **cuDNN** - Deep learning primitives
