/**
 * Simple Audio Filter - Time Domain Low-Pass Filter
 * 
 * This demonstrates a simple moving average filter implemented on GPU.
 * While real audio plugins use FFT-based filtering or IIR filters,
 * this is a great starting point for understanding parallel audio processing.
 * 
 * A moving average filter is a simple low-pass filter that smooths the signal.
 * 
 * Compile: nvcc -o simple_filter simple_filter.cu
 * Run: ./simple_filter
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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
 * CUDA Kernel: Moving Average Filter (Simple Low-Pass)
 * 
 * Averages windowSize samples centered around each sample.
 * Larger window = more smoothing = lower cutoff frequency
 * 
 * NOTE: This is a FIR filter with equal coefficients (box filter)
 */
__global__ void movingAverageFilter(const float *input, float *output, 
                                     int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        int halfWindow = windowSize / 2;
        int count = 0;
        
        // Sum samples in the window
        for (int i = -halfWindow; i <= halfWindow; i++) {
            int sampleIdx = idx + i;
            // Boundary handling: clamp to valid range
            if (sampleIdx >= 0 && sampleIdx < n) {
                sum += input[sampleIdx];
                count++;
            }
        }
        
        output[idx] = sum / count;
    }
}

/**
 * CUDA Kernel: One-Pole Low-Pass Filter (IIR)
 * 
 * A more musical low-pass filter using a simple recursive formula:
 * y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
 * 
 * alpha = cutoff frequency control (0 to 1)
 * Lower alpha = more filtering (darker sound)
 * 
 * LIMITATION: This kernel processes sequentially due to data dependency.
 * In practice, you'd process multiple channels in parallel instead.
 */
__global__ void onePoleFilter(const float *input, float *output, 
                               int n, float alpha) {
    // Note: This is inherently sequential per-channel
    // We parallelize across channels, not samples
    // For this demo, we process the single channel
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = input[0];  // Initialize
        
        for (int i = 1; i < n; i++) {
            output[i] = alpha * input[i] + (1.0f - alpha) * output[i-1];
        }
    }
}

/**
 * CUDA Kernel: Parallel One-Pole for Multiple Channels
 * 
 * Better use of GPU: process multiple audio channels simultaneously
 * Each thread handles one complete channel
 */
__global__ void onePoleFilterMultiChannel(const float *input, float *output,
                                           int samplesPerChannel, int numChannels,
                                           float alpha) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (channel < numChannels) {
        int offset = channel * samplesPerChannel;
        
        output[offset] = input[offset];  // Initialize first sample
        
        // Process all samples in this channel
        for (int i = 1; i < samplesPerChannel; i++) {
            output[offset + i] = alpha * input[offset + i] + 
                                 (1.0f - alpha) * output[offset + i - 1];
        }
    }
}

/**
 * CUDA Kernel: Simple High-Pass Filter
 * 
 * High-pass = original - low-pass
 * This removes low frequencies, keeping high frequencies.
 */
__global__ void highPassFromLowPass(const float *original, const float *lowPassed,
                                     float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = original[idx] - lowPassed[idx];
    }
}

/**
 * Generate test signals
 */
void generateMixedSignal(float *samples, int n) {
    // Mix of frequencies to test filtering:
    // 100 Hz (bass), 440 Hz (mid), 2000 Hz (high)
    const float kPi = 3.14159265358979323846f;
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        samples[i] = 0.5f * sinf(2.0f * kPi * 100.0f * t) +   // Bass
                     0.3f * sinf(2.0f * kPi * 440.0f * t) +   // Mid
                     0.2f * sinf(2.0f * kPi * 2000.0f * t);   // High
    }
}

void generateWhiteNoise(float *samples, int n, float amplitude) {
    for (int i = 0; i < n; i++) {
        samples[i] = amplitude * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    }
}

/**
 * Estimate dominant frequency using zero-crossing rate
 * (Simple approximation - real analysis would use FFT)
 */
float estimateZeroCrossingRate(const float *samples, int n) {
    int crossings = 0;
    for (int i = 1; i < n; i++) {
        if ((samples[i-1] >= 0 && samples[i] < 0) ||
            (samples[i-1] < 0 && samples[i] >= 0)) {
            crossings++;
        }
    }
    // Zero-crossing rate approximates 2x the fundamental frequency
    return (float)crossings * SAMPLE_RATE / (2.0f * n);
}

float calculateRMS(const float *samples, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += samples[i] * samples[i];
    }
    return sqrtf(sum / n);
}

float calculatePeak(const float *samples, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; i++) {
        float absVal = fabsf(samples[i]);
        if (absVal > peak) peak = absVal;
    }
    return peak;
}

int main() {
    printf("=== CUDA Audio Filter Demo ===\n");
    printf("Processing %d samples at %d Hz\n\n", NUM_SAMPLES, SAMPLE_RATE);
    
    size_t bytes = NUM_SAMPLES * sizeof(float);
    
    // Allocate memory
    float *h_input = (float*)malloc(bytes);
    float *h_filtered = (float*)malloc(bytes);
    float *h_highpass = (float*)malloc(bytes);
    if (!h_input || !h_filtered || !h_highpass) {
        fprintf(stderr, "Failed to allocate host memory\n");
        free(h_input);
        free(h_filtered);
        free(h_highpass);
        return EXIT_FAILURE;
    }
    
    // ========================================
    // Test with mixed frequency signal
    // ========================================
    printf("[1] Generating test signal (100 Hz + 440 Hz + 2000 Hz)...\n");
    generateMixedSignal(h_input, NUM_SAMPLES);
    
    printf("    Original: Peak=%.3f, RMS=%.3f, Est.ZCR=%.1f Hz\n",
           calculatePeak(h_input, NUM_SAMPLES),
           calculateRMS(h_input, NUM_SAMPLES),
           estimateZeroCrossingRate(h_input, NUM_SAMPLES));
    
    // Allocate device memory
    printf("\n[2] Allocating GPU memory...\n");
    float *d_input, *d_filtered, *d_highpass;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_filtered, bytes));
    CUDA_CHECK(cudaMalloc(&d_highpass, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_SAMPLES + threadsPerBlock - 1) / threadsPerBlock;
    
    // ========================================
    // Test: Moving Average Filters
    // ========================================
    printf("\n[3] Testing Moving Average Low-Pass Filters...\n");
    
    int windowSizes[] = {5, 21, 101, 501};
    int numTests = 4;
    
    for (int t = 0; t < numTests; t++) {
        int windowSize = windowSizes[t];
        
        movingAverageFilter<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_filtered, NUM_SAMPLES, windowSize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_filtered, d_filtered, bytes, cudaMemcpyDeviceToHost));
        
        printf("    Window=%3d: Peak=%.3f, RMS=%.3f, Est.ZCR=%.1f Hz\n",
               windowSize,
               calculatePeak(h_filtered, NUM_SAMPLES),
               calculateRMS(h_filtered, NUM_SAMPLES),
               estimateZeroCrossingRate(h_filtered, NUM_SAMPLES));
    }
    printf("    (Larger window = more low-pass filtering)\n");
    
    // ========================================
    // Test: One-Pole Filter
    // ========================================
    printf("\n[4] Testing One-Pole Low-Pass Filter...\n");
    
    float alphas[] = {1.0f, 0.5f, 0.1f, 0.01f};
    int numAlphas = 4;
    
    for (int t = 0; t < numAlphas; t++) {
        float alpha = alphas[t];
        
        onePoleFilter<<<1, 1>>>(d_input, d_filtered, NUM_SAMPLES, alpha);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_filtered, d_filtered, bytes, cudaMemcpyDeviceToHost));
        
        printf("    Alpha=%.2f: Peak=%.3f, RMS=%.3f, Est.ZCR=%.1f Hz\n",
               alpha,
               calculatePeak(h_filtered, NUM_SAMPLES),
               calculateRMS(h_filtered, NUM_SAMPLES),
               estimateZeroCrossingRate(h_filtered, NUM_SAMPLES));
    }
    printf("    (Lower alpha = more filtering, darker sound)\n");
    
    // ========================================
    // Test: High-Pass via Subtraction
    // ========================================
    printf("\n[5] Testing High-Pass Filter (Original - LowPass)...\n");
    
    // Apply strong low-pass first
    movingAverageFilter<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_filtered, NUM_SAMPLES, 101
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Subtract to get high-pass
    highPassFromLowPass<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_filtered, d_highpass, NUM_SAMPLES
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_highpass, d_highpass, bytes, cudaMemcpyDeviceToHost));
    
    printf("    High-pass result: Peak=%.3f, RMS=%.3f, Est.ZCR=%.1f Hz\n",
           calculatePeak(h_highpass, NUM_SAMPLES),
           calculateRMS(h_highpass, NUM_SAMPLES),
           estimateZeroCrossingRate(h_highpass, NUM_SAMPLES));
    printf("    (Higher ZCR = more high-frequency content preserved)\n");
    
    // ========================================
    // Test: Filter Noise
    // ========================================
    printf("\n[6] Testing noise filtering...\n");
    generateWhiteNoise(h_input, NUM_SAMPLES, 0.5f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    printf("    White noise: Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_input, NUM_SAMPLES),
           calculateRMS(h_input, NUM_SAMPLES));
    
    // Apply heavy filtering
    movingAverageFilter<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_filtered, NUM_SAMPLES, 201
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_filtered, d_filtered, bytes, cudaMemcpyDeviceToHost));
    
    printf("    Filtered:    Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_filtered, NUM_SAMPLES),
           calculateRMS(h_filtered, NUM_SAMPLES));
    printf("    (Reduced RMS indicates noise attenuation)\n");
    
    // Cleanup
    printf("\n[7] Cleaning up...\n");
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filtered));
    CUDA_CHECK(cudaFree(d_highpass));
    free(h_input);
    free(h_filtered);
    free(h_highpass);
    
    printf("\n=== Filter Concepts Demonstrated ===\n");
    printf("  • Moving average (FIR) low-pass filter\n");
    printf("  • One-pole (IIR) recursive filter\n");
    printf("  • High-pass via spectral subtraction\n");
    printf("  • Parallel vs sequential processing trade-offs\n");
    printf("\nNext steps: Implement proper FFT-based filtering with cuFFT!\n");
    
    return EXIT_SUCCESS;
}
