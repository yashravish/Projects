/**
 * Pitch detector interface and types
 */

import type { Hz } from "../../types";

/**
 * Result of a single pitch detection
 */
export interface PitchDetectionResult {
  /** Detected frequency in Hz, or null if unvoiced */
  f0: Hz | null;
  /** Voicing confidence 0..1 */
  voicing: number;
}

/**
 * Pitch detector interface
 */
export interface PitchDetector {
  /**
   * Detect pitch in a windowed audio frame
   * @param frame Audio samples (should be windowed)
   * @param sampleRate Sample rate in Hz
   * @returns Pitch detection result
   */
  detect(frame: Float32Array, sampleRate: number): PitchDetectionResult;
}

/**
 * Configuration for pitch detection
 */
export interface PitchDetectorConfig {
  /** Minimum frequency to detect in Hz */
  minFreq?: number;
  /** Maximum frequency to detect in Hz */
  maxFreq?: number;
  /** Threshold for voicing detection */
  voicingThreshold?: number;
}
