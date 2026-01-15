/**
 * Vector Addition - The "Hello World" of CUDA
 * 
 * This example demonstrates the fundamental CUDA workflow:
 * 1. Allocate memory on host (CPU) and device (GPU)
 * 2. Copy data from host to device
 * 3. Execute parallel computation on GPU
 * 4. Copy results back to host
 * 5. Free memory
 * 
 * Compile: nvcc -o vector_add vector_add.cu
 * Run: ./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Number of elements in our vectors
#define N 1000000

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * CUDA Kernel: Vector Addition
 * 
 * __global__ - This function runs on the GPU but is called from the CPU
 * 
 * Each thread computes ONE element of the output vector.
 * With N=1,000,000 elements, we launch ~1 million threads!
 */
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate unique global thread ID
    // blockIdx.x  = which block this thread belongs to
    // blockDim.x  = number of threads per block
    // threadIdx.x = thread's index within its block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check: don't access memory outside our arrays
    // (We might launch more threads than we have elements)
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * CPU version for comparison
 */
void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Verify GPU results match CPU results
 */
int verifyResults(const float *gpu_result, const float *cpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", 
                   i, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    printf("=== CUDA Vector Addition Demo ===\n");
    printf("Vector size: %d elements\n\n", N);
    
    // Calculate memory size needed
    size_t bytes = N * sizeof(float);
    
    // ========================================
    // Step 1: Allocate HOST (CPU) memory
    // ========================================
    printf("[1] Allocating host memory...\n");
    float *h_a = (float*)malloc(bytes);  // Input vector A
    float *h_b = (float*)malloc(bytes);  // Input vector B
    float *h_c = (float*)malloc(bytes);  // Output vector (GPU result)
    float *h_c_cpu = (float*)malloc(bytes);  // Output (CPU result for verification)
    
    if (!h_a || !h_b || !h_c || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input vectors with sample data
    printf("[2] Initializing vectors...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i) * sinf(i);    // Some interesting values
        h_b[i] = cosf(i) * cosf(i);
    }
    
    // ========================================
    // Step 2: Allocate DEVICE (GPU) memory
    // ========================================
    printf("[3] Allocating device memory...\n");
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // ========================================
    // Step 3: Copy data from HOST to DEVICE
    // ========================================
    printf("[4] Copying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // ========================================
    // Step 4: Configure and launch the kernel
    // ========================================
    printf("[5] Launching GPU kernel...\n");
    
    // Thread configuration:
    // - Each thread block has a max of 1024 threads (hardware limit)
    // - We choose 256 threads per block (good balance)
    // - We need enough blocks to cover all N elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("    Grid: %d blocks, %d threads/block = %d total threads\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);
    
    // Launch the kernel!
    // <<<blocksPerGrid, threadsPerBlock>>> is CUDA's execution configuration
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ========================================
    // Step 5: Copy results from DEVICE to HOST
    // ========================================
    printf("[6] Copying results back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // ========================================
    // Verification: Compare with CPU result
    // ========================================
    printf("[7] Running CPU version for verification...\n");
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    
    if (verifyResults(h_c, h_c_cpu, N)) {
        printf("\nSUCCESS: GPU results match CPU results!\n");
    } else {
        printf("\nFAILURE: Results don't match!\n");
    }
    
    // Print a few sample results
    printf("\nSample results (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  a[%d] + b[%d] = %.6f + %.6f = %.6f\n",
               i, i, h_a[i], h_b[i], h_c[i]);
    }
    
    // ========================================
    // Step 6: Free memory
    // ========================================
    printf("\n[8] Cleaning up...\n");
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    
    printf("\nDone! Key concepts demonstrated:\n");
    printf("  • __global__ kernel function\n");
    printf("  • Thread indexing with blockIdx, blockDim, threadIdx\n");
    printf("  • cudaMalloc, cudaMemcpy, cudaFree\n");
    printf("  • <<<blocks, threads>>> launch configuration\n");
    
    return EXIT_SUCCESS;
}
