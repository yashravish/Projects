/**
 * Audio Gain Control - GPU-Accelerated Volume Adjustment
 * 
 * This example applies what we learned in vector_add to a real audio use case:
 * adjusting the volume (gain) of audio samples in parallel.
 * 
 * In audio processing:
 * - Samples are typically 32-bit floats in range [-1.0, 1.0]
 * - At 44.1kHz sample rate, 1 second = 44,100 samples
 * - GPU parallelism shines when processing large audio buffers
 * 
 * Compile: nvcc -o audio_gain audio_gain.cu
 * Run: ./audio_gain
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Simulate 1 second of audio at 44.1kHz
#define SAMPLE_RATE 44100
#define DURATION_SECONDS 1
#define NUM_SAMPLES (SAMPLE_RATE * DURATION_SECONDS)

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
 * CUDA Kernel: Apply Gain to Audio Samples
 * 
 * Each thread processes ONE audio sample - perfect parallelization!
 * 
 * @param samples  Audio sample buffer (modified in-place)
 * @param gain     Volume multiplier (0.0 = silence, 1.0 = original, 2.0 = double)
 * @param n        Number of samples
 */
__global__ void applyGain(float *samples, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Apply gain
        float result = samples[idx] * gain;
        
        // Clip to valid audio range [-1.0, 1.0] to prevent distortion
        // This is called "hard clipping" - a simple form of limiting
        if (result > 1.0f) result = 1.0f;
        if (result < -1.0f) result = -1.0f;
        
        samples[idx] = result;
    }
}

/**
 * CUDA Kernel: Apply Gain with Soft Clipping (Tanh Saturation)
 * 
 * This produces a more musical distortion when signal exceeds range.
 * Used in guitar amp simulations and warm-sounding limiters.
 */
__global__ void applyGainSoftClip(float *samples, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Apply gain
        float result = samples[idx] * gain;
        
        // Soft clipping using tanh - compresses rather than cuts
        // This gives a warmer, more analog-like saturation
        result = tanhf(result);
        
        samples[idx] = result;
    }
}

/**
 * Generate a simple sine wave (simulating a musical note)
 * 
 * @param samples   Output buffer
 * @param n         Number of samples
 * @param freq      Frequency in Hz (e.g., 440 = A4 note)
 * @param amplitude Initial amplitude (0.0 to 1.0)
 */
void generateSineWave(float *samples, int n, float freq, float amplitude) {
    const float kPi = 3.14159265358979323846f;
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        samples[i] = amplitude * sinf(2.0f * kPi * freq * t);
    }
}

/**
 * Calculate RMS (Root Mean Square) level - common audio loudness metric
 */
float calculateRMS(const float *samples, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += samples[i] * samples[i];
    }
    return sqrtf(sum / n);
}

/**
 * Calculate peak level (maximum absolute value)
 */
float calculatePeak(const float *samples, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; i++) {
        float absVal = fabsf(samples[i]);
        if (absVal > peak) peak = absVal;
    }
    return peak;
}

int main() {
    printf("=== CUDA Audio Gain Control Demo ===\n");
    printf("Simulating %d samples (%d second at %d Hz)\n\n", 
           NUM_SAMPLES, DURATION_SECONDS, SAMPLE_RATE);
    
    size_t bytes = NUM_SAMPLES * sizeof(float);
    
    // Allocate host memory
    float *h_samples = (float*)malloc(bytes);
    float *h_original = (float*)malloc(bytes);  // Keep original for comparison
    
    if (!h_samples || !h_original) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Generate a 440Hz sine wave (the A4 note) at 50% amplitude
    printf("[1] Generating 440Hz sine wave (A4 note)...\n");
    generateSineWave(h_samples, NUM_SAMPLES, 440.0f, 0.5f);
    memcpy(h_original, h_samples, bytes);
    
    printf("    Original signal: Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_samples, NUM_SAMPLES),
           calculateRMS(h_samples, NUM_SAMPLES));
    
    // Allocate device memory
    printf("\n[2] Allocating GPU memory...\n");
    float *d_samples;
    CUDA_CHECK(cudaMalloc(&d_samples, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_samples, h_samples, bytes, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_SAMPLES + threadsPerBlock - 1) / threadsPerBlock;
    
    // ========================================
    // Test 1: Reduce volume (gain < 1.0)
    // ========================================
    printf("\n[3] Test 1: Applying 0.5x gain (reduce volume by half)...\n");
    
    float gain = 0.5f;
    applyGain<<<blocksPerGrid, threadsPerBlock>>>(d_samples, gain, NUM_SAMPLES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_samples, d_samples, bytes, cudaMemcpyDeviceToHost));
    
    printf("    After 0.5x gain: Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_samples, NUM_SAMPLES),
           calculateRMS(h_samples, NUM_SAMPLES));
    
    // ========================================
    // Test 2: Boost with hard clipping
    // ========================================
    printf("\n[4] Test 2: Applying 3.0x gain with HARD clipping...\n");
    
    // Reset to original
    CUDA_CHECK(cudaMemcpy(d_samples, h_original, bytes, cudaMemcpyHostToDevice));
    
    gain = 3.0f;
    applyGain<<<blocksPerGrid, threadsPerBlock>>>(d_samples, gain, NUM_SAMPLES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_samples, d_samples, bytes, cudaMemcpyDeviceToHost));
    
    printf("    After 3.0x gain (hard clip): Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_samples, NUM_SAMPLES),
           calculateRMS(h_samples, NUM_SAMPLES));
    printf("    Note: Peak is capped at 1.0 (clipping occurred!)\n");
    
    // ========================================
    // Test 3: Boost with soft clipping (tanh)
    // ========================================
    printf("\n[5] Test 3: Applying 3.0x gain with SOFT clipping (tanh)...\n");
    
    // Reset to original
    CUDA_CHECK(cudaMemcpy(d_samples, h_original, bytes, cudaMemcpyHostToDevice));
    
    applyGainSoftClip<<<blocksPerGrid, threadsPerBlock>>>(d_samples, 3.0f, NUM_SAMPLES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_samples, d_samples, bytes, cudaMemcpyDeviceToHost));
    
    printf("    After 3.0x gain (soft clip): Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_samples, NUM_SAMPLES),
           calculateRMS(h_samples, NUM_SAMPLES));
    printf("    Note: Soft clipping preserves more signal shape (warmer sound)\n");
    
    // Cleanup
    printf("\n[6] Cleaning up...\n");
    CUDA_CHECK(cudaFree(d_samples));
    free(h_samples);
    free(h_original);
    
    printf("\n=== Audio Processing Concepts Demonstrated ===\n");
    printf("  • In-place buffer modification on GPU\n");
    printf("  • Hard clipping (digital limiting)\n");
    printf("  • Soft clipping (analog-style saturation)\n");
    printf("  • Audio metrics: Peak level, RMS level\n");
    printf("\nThis is the foundation for real-time audio effects!\n");
    
    return EXIT_SUCCESS;
}
