/**
 * Autocorrelation Function (ACF) pitch detection
 * Simpler and faster than YIN, but less robust
 */

import type { PitchDetector, PitchDetectionResult, PitchDetectorConfig } from "./types";

export class ACFDetector implements PitchDetector {
  private minFreq: number;
  private maxFreq: number;
  private voicingThreshold: number;

  constructor(config: PitchDetectorConfig = {}) {
    this.minFreq = config.minFreq ?? 80; // Hz
    this.maxFreq = config.maxFreq ?? 800; // Hz
    this.voicingThreshold = config.voicingThreshold ?? 0.3;
  }

  detect(frame: Float32Array, sampleRate: number): PitchDetectionResult {
    const minPeriod = Math.floor(sampleRate / this.maxFreq);
    const maxPeriod = Math.floor(sampleRate / this.minFreq);

    // Guard against silent frames
    const rms = Math.sqrt(
      frame.reduce((sum, val) => sum + val * val, 0) / frame.length
    );
    if (rms < 1e-6) {
      return { f0: null, voicing: 0 };
    }

    // Calculate autocorrelation
    const acf = this.autocorrelation(frame, maxPeriod);

    // Normalize by the zero-lag value
    const r0 = acf[0];
    if (r0 === 0) {
      return { f0: null, voicing: 0 };
    }

    for (let i = 0; i < acf.length; i++) {
      acf[i] /= r0;
    }

    // Find the first peak after minPeriod
    const tau = this.findFirstPeak(acf, minPeriod, maxPeriod);

    if (tau === -1 || acf[tau] < this.voicingThreshold) {
      return { f0: null, voicing: 0 };
    }

    // Parabolic interpolation for better precision
    const period = this.parabolicInterpolation(acf, tau);

    return {
      f0: sampleRate / period,
      voicing: Math.max(0, Math.min(1, acf[tau])),
    };
  }

  /**
   * Calculate autocorrelation function
   */
  private autocorrelation(buffer: Float32Array, maxLag: number): Float32Array {
    const acf = new Float32Array(maxLag);
    const n = buffer.length;

    for (let lag = 0; lag < maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < n - lag; i++) {
        sum += buffer[i] * buffer[i + lag];
      }
      acf[lag] = sum;
    }

    return acf;
  }

  /**
   * Find the first significant peak in the ACF
   */
  private findFirstPeak(acf: Float32Array, minLag: number, maxLag: number): number {
    let maxValue = -Infinity;
    let maxIndex = -1;

    for (let i = minLag; i < Math.min(maxLag, acf.length); i++) {
      // Check if it's a local maximum
      if (i > 0 && i < acf.length - 1) {
        if (acf[i] > acf[i - 1] && acf[i] > acf[i + 1] && acf[i] > maxValue) {
          maxValue = acf[i];
          maxIndex = i;
        }
      }
    }

    return maxIndex;
  }

  /**
   * Parabolic interpolation for sub-sample precision
   */
  private parabolicInterpolation(acf: Float32Array, tau: number): number {
    if (tau === 0 || tau >= acf.length - 1) {
      return tau;
    }

    const alpha = acf[tau - 1];
    const beta = acf[tau];
    const gamma = acf[tau + 1];

    // Parabolic peak location
    const p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);

    return tau + p;
  }
}
