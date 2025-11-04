/**
 * YIN pitch detection algorithm
 * Based on: Cheveigné, A. de, & Kawahara, H. (2002)
 * "YIN, a fundamental frequency estimator for speech and music"
 */

import type { PitchDetector, PitchDetectionResult, PitchDetectorConfig } from "./types";

export class YINDetector implements PitchDetector {
  private minFreq: number;
  private maxFreq: number;
  private voicingThreshold: number;

  constructor(config: PitchDetectorConfig = {}) {
    this.minFreq = config.minFreq ?? 80; // Hz
    this.maxFreq = config.maxFreq ?? 800; // Hz
    this.voicingThreshold = config.voicingThreshold ?? 0.15;
  }

  detect(frame: Float32Array, sampleRate: number): PitchDetectionResult {
    const minPeriod = Math.floor(sampleRate / this.maxFreq);
    const maxPeriod = Math.floor(sampleRate / this.minFreq);

    // Guard against silent frames to avoid false voicing
    const rms = Math.sqrt(
      frame.reduce((sum, val) => sum + val * val, 0) / frame.length
    );
    if (rms < 1e-6) {
      return { f0: null, voicing: 0 };
    }

    // Step 1: Calculate difference function
    const diff = this.differenceFunction(frame, maxPeriod);

    // Step 2: Cumulative mean normalized difference
    const cmnd = this.cumulativeMeanNormalizedDifference(diff);

    // Step 3: Absolute threshold
    const tau = this.absoluteThreshold(cmnd, minPeriod, maxPeriod);

    if (tau === -1) {
      return { f0: null, voicing: 0 };
    }

    // Step 4: Parabolic interpolation for better precision
    const period = this.parabolicInterpolation(cmnd, tau);

    // Bounds check on period
    if (period < minPeriod || period > maxPeriod) {
      return { f0: null, voicing: 0 };
    }

    // Calculate voicing confidence (inverse of CMND value)
    const voicing = 1 - cmnd[tau];

    return {
      f0: sampleRate / period,
      voicing: Math.max(0, Math.min(1, voicing)),
    };
  }

  /**
   * Calculate the difference function (autocorrelation-like)
   */
  private differenceFunction(buffer: Float32Array, maxPeriod: number): Float32Array {
    const diff = new Float32Array(maxPeriod);

    for (let tau = 0; tau < maxPeriod; tau++) {
      let sum = 0;
      for (let i = 0; i < buffer.length - maxPeriod; i++) {
        const delta = buffer[i] - buffer[i + tau];
        sum += delta * delta;
      }
      diff[tau] = sum;
    }

    return diff;
  }

  /**
   * Cumulative mean normalized difference function
   */
  private cumulativeMeanNormalizedDifference(diff: Float32Array): Float32Array {
    const cmnd = new Float32Array(diff.length);
    cmnd[0] = 1;

    let runningSum = 0;

    for (let tau = 1; tau < diff.length; tau++) {
      runningSum += diff[tau];
      cmnd[tau] = diff[tau] / (runningSum / tau);
    }

    return cmnd;
  }

  /**
   * Find the first minimum below threshold
   */
  private absoluteThreshold(cmnd: Float32Array, minPeriod: number, maxPeriod: number): number {
    // Start searching from minPeriod
    for (let tau = minPeriod; tau < maxPeriod; tau++) {
      if (cmnd[tau] < this.voicingThreshold) {
        // Look for local minimum
        while (tau + 1 < maxPeriod && cmnd[tau + 1] < cmnd[tau]) {
          tau++;
        }
        return tau;
      }
    }

    // No period found
    return -1;
  }

  /**
   * Parabolic interpolation for sub-sample precision
   */
  private parabolicInterpolation(cmnd: Float32Array, tau: number): number {
    if (tau === 0 || tau >= cmnd.length - 1) {
      return tau;
    }

    const s0 = cmnd[tau - 1];
    const s1 = cmnd[tau];
    const s2 = cmnd[tau + 1];

    // Parabolic interpolation formula - protect against divide by zero
    const denominator = 2 * (2 * s1 - s0 - s2);
    if (Math.abs(denominator) < 1e-10) {
      return tau;
    }

    const adjustment = (s0 - s2) / denominator;

    return tau + adjustment;
  }
}
