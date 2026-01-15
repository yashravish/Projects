/**
 * Audio Mixer - GPU-Accelerated Signal Mixing
 * 
 * Mix multiple audio tracks together with individual volume controls.
 * This is fundamental to digital audio workstations (DAWs) and
 * live sound processing.
 * 
 * Use case: Mixing multiple instruments/vocals into a stereo output
 * 
 * Compile: nvcc -o audio_mixer audio_mixer.cu
 * Run: ./audio_mixer
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define SAMPLE_RATE 44100
#define DURATION_SECONDS 1
#define NUM_SAMPLES (SAMPLE_RATE * DURATION_SECONDS)
#define NUM_TRACKS 4  // Mix 4 audio tracks

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
 * CUDA Kernel: Mix Two Audio Tracks
 * 
 * Simple 2-track mixer with individual gains
 */
__global__ void mixTwoTracks(const float *track1, const float *track2,
                             float gain1, float gain2,
                             float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Mix with gains, then normalize to prevent clipping
        float mixed = (track1[idx] * gain1) + (track2[idx] * gain2);
        
        // Soft clip the output
        output[idx] = tanhf(mixed);
    }
}

/**
 * CUDA Kernel: Mix Four Audio Tracks
 * 
 * More realistic: mixing multiple instruments together
 * Each track has its own gain control
 */
__global__ void mixFourTracks(const float *track1, const float *track2,
                              const float *track3, const float *track4,
                              float gain1, float gain2, float gain3, float gain4,
                              float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float mixed = (track1[idx] * gain1) + 
                      (track2[idx] * gain2) +
                      (track3[idx] * gain3) +
                      (track4[idx] * gain4);
        
        // Master soft limiter
        output[idx] = tanhf(mixed * 0.5f);  // 0.5 = master gain reduction
    }
}

/**
 * CUDA Kernel: Crossfade Between Two Tracks
 * 
 * Smoothly transition from one track to another
 * Essential for DJ software and audio editing
 */
__global__ void crossfade(const float *trackA, const float *trackB,
                          float *output, int n, float fadePosition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // fadePosition: 0.0 = all trackA, 1.0 = all trackB
        // Using equal-power crossfade for smooth transitions
        const float kPi = 3.14159265358979323846f;
        float gainA = cosf(fadePosition * kPi * 0.5f);
        float gainB = sinf(fadePosition * kPi * 0.5f);
        
        output[idx] = (trackA[idx] * gainA) + (trackB[idx] * gainB);
    }
}

/**
 * Generate different waveforms for testing
 */
void generateSineWave(float *samples, int n, float freq, float amp) {
    const float kPi = 3.14159265358979323846f;
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        samples[i] = amp * sinf(2.0f * kPi * freq * t);
    }
}

void generateSquareWave(float *samples, int n, float freq, float amp) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float phase = fmodf(t * freq, 1.0f);
        samples[i] = amp * (phase < 0.5f ? 1.0f : -1.0f);
    }
}

void generateSawtoothWave(float *samples, int n, float freq, float amp) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float phase = fmodf(t * freq, 1.0f);
        samples[i] = amp * (2.0f * phase - 1.0f);
    }
}

void generateTriangleWave(float *samples, int n, float freq, float amp) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float phase = fmodf(t * freq, 1.0f);
        samples[i] = amp * (4.0f * fabsf(phase - 0.5f) - 1.0f);
    }
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
    printf("=== CUDA Audio Mixer Demo ===\n");
    printf("Mixing %d tracks, %d samples each\n\n", NUM_TRACKS, NUM_SAMPLES);
    
    size_t bytes = NUM_SAMPLES * sizeof(float);
    
    // ========================================
    // Allocate and generate test signals
    // ========================================
    printf("[1] Generating test signals...\n");
    
    // Host memory for 4 tracks + output
    float *h_tracks[NUM_TRACKS] = {0};
    float *h_output = (float*)malloc(bytes);
    
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < NUM_TRACKS; i++) {
        h_tracks[i] = (float*)malloc(bytes);
        if (!h_tracks[i]) {
            fprintf(stderr, "Failed to allocate host memory for track %d\n", i + 1);
            for (int j = 0; j < i; j++) {
                free(h_tracks[j]);
            }
            free(h_output);
            return EXIT_FAILURE;
        }
    }
    
    // Generate different waveforms at different frequencies
    // Simulating a simple musical mix:
    // Track 1: Bass (low sine wave) - 110 Hz (A2)
    // Track 2: Lead melody (sawtooth) - 440 Hz (A4)
    // Track 3: Pad (triangle) - 220 Hz (A3)
    // Track 4: Percussion-like (square) - 880 Hz (A5)
    
    printf("    Track 1: Bass sine wave at 110 Hz\n");
    generateSineWave(h_tracks[0], NUM_SAMPLES, 110.0f, 0.6f);
    
    printf("    Track 2: Lead sawtooth at 440 Hz\n");
    generateSawtoothWave(h_tracks[1], NUM_SAMPLES, 440.0f, 0.4f);
    
    printf("    Track 3: Pad triangle at 220 Hz\n");
    generateTriangleWave(h_tracks[2], NUM_SAMPLES, 220.0f, 0.3f);
    
    printf("    Track 4: High square at 880 Hz\n");
    generateSquareWave(h_tracks[3], NUM_SAMPLES, 880.0f, 0.2f);
    
    // Print individual track levels
    printf("\nIndividual track levels:\n");
    for (int i = 0; i < NUM_TRACKS; i++) {
        printf("    Track %d: Peak=%.3f, RMS=%.3f\n", 
               i+1, calculatePeak(h_tracks[i], NUM_SAMPLES),
               calculateRMS(h_tracks[i], NUM_SAMPLES));
    }
    
    // ========================================
    // Allocate device memory
    // ========================================
    printf("\n[2] Allocating GPU memory...\n");
    
    float *d_tracks[NUM_TRACKS];
    float *d_output;
    
    for (int i = 0; i < NUM_TRACKS; i++) {
        CUDA_CHECK(cudaMalloc(&d_tracks[i], bytes));
        CUDA_CHECK(cudaMemcpy(d_tracks[i], h_tracks[i], bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_SAMPLES + threadsPerBlock - 1) / threadsPerBlock;
    
    // ========================================
    // Test 1: Mix all 4 tracks
    // ========================================
    printf("\n[3] Test 1: Mixing all 4 tracks...\n");
    printf("    Gains: Bass=0.8, Lead=0.6, Pad=0.5, High=0.3\n");
    
    mixFourTracks<<<blocksPerGrid, threadsPerBlock>>>(
        d_tracks[0], d_tracks[1], d_tracks[2], d_tracks[3],
        0.8f, 0.6f, 0.5f, 0.3f,  // Individual track gains
        d_output, NUM_SAMPLES
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    printf("    Mixed output: Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_output, NUM_SAMPLES),
           calculateRMS(h_output, NUM_SAMPLES));
    
    // ========================================
    // Test 2: Crossfade demonstration
    // ========================================
    printf("\n[4] Test 2: Crossfade between Track 1 and Track 2...\n");
    
    float fadePositions[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    int numFades = 5;
    
    for (int i = 0; i < numFades; i++) {
        float fade = fadePositions[i];
        
        crossfade<<<blocksPerGrid, threadsPerBlock>>>(
            d_tracks[0], d_tracks[1], d_output, NUM_SAMPLES, fade
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
        
        printf("    Fade=%.2f: Peak=%.3f (%.0f%% Track1, %.0f%% Track2)\n",
               fade, calculatePeak(h_output, NUM_SAMPLES),
               (1.0f - fade) * 100, fade * 100);
    }
    
    // ========================================
    // Test 3: Simple 2-track mix
    // ========================================
    printf("\n[5] Test 3: Two-track mix (Bass + Lead)...\n");
    
    mixTwoTracks<<<blocksPerGrid, threadsPerBlock>>>(
        d_tracks[0], d_tracks[1],
        1.0f, 0.7f,
        d_output, NUM_SAMPLES
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    printf("    Bass(1.0) + Lead(0.7): Peak=%.3f, RMS=%.3f\n",
           calculatePeak(h_output, NUM_SAMPLES),
           calculateRMS(h_output, NUM_SAMPLES));
    
    // ========================================
    // Cleanup
    // ========================================
    printf("\n[6] Cleaning up...\n");
    
    for (int i = 0; i < NUM_TRACKS; i++) {
        CUDA_CHECK(cudaFree(d_tracks[i]));
        free(h_tracks[i]);
    }
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);
    
    printf("\n=== Mixing Concepts Demonstrated ===\n");
    printf("  • Multi-track mixing with individual gains\n");
    printf("  • Equal-power crossfade for smooth transitions\n");
    printf("  • Master bus processing (soft limiting)\n");
    printf("  • Different waveform generation\n");
    printf("\nFoundation for building a GPU-accelerated DAW!\n");
    
    return EXIT_SUCCESS;
}
