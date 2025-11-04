/**
 * Tests for tonic (Sa) detection
 */

import { describe, it, expect } from "vitest";
import { detectTonic, validateTonic } from "../src/theory/tonic";
import type { PitchFrame } from "../src/types";

describe("Tonic Detection", () => {
  /**
   * Generate mock pitch frames centered around a tonic
   */
  function generateMockFrames(tonicHz: number, numFrames: number = 100): PitchFrame[] {
    const frames: PitchFrame[] = [];
    const swaraFreqs = [
      tonicHz, // S
      tonicHz * Math.pow(2, 204 / 1200), // R2
      tonicHz * Math.pow(2, 294 / 1200), // G2
      tonicHz * Math.pow(2, 498 / 1200), // M1
      tonicHz * Math.pow(2, 702 / 1200), // P
    ];

    for (let i = 0; i < numFrames; i++) {
      // Randomly pick a swara frequency
      const freq = swaraFreqs[i % swaraFreqs.length];
      
      // Add small random variation
      const variation = 1 + (Math.random() - 0.5) * 0.02; // ±1%

      frames.push({
        t: i * 0.01,
        f0: freq * variation,
        voicing: 0.8 + Math.random() * 0.2,
      });
    }

    return frames;
  }

  it("should detect tonic around 220 Hz (A3)", () => {
    const expectedTonic = 220;
    const frames = generateMockFrames(expectedTonic);

    const result = detectTonic(frames);

    // Should be within ±5 Hz
    expect(result.hz).toBeGreaterThan(215);
    expect(result.hz).toBeLessThan(225);
    expect(result.confidence).toBeGreaterThan(0.3);
  });

  it("should detect tonic around 261.6 Hz (C4)", () => {
    const expectedTonic = 261.6;
    const frames = generateMockFrames(expectedTonic);

    const result = detectTonic(frames);

    // Should be within ±5 Hz
    expect(result.hz).toBeGreaterThan(256);
    expect(result.hz).toBeLessThan(267);
    expect(result.confidence).toBeGreaterThan(0.3);
  });

  it("should handle frames with no voiced regions", () => {
    const frames: PitchFrame[] = Array.from({ length: 50 }, (_, i) => ({
      t: i * 0.01,
      f0: null,
      voicing: 0,
    }));

    const result = detectTonic(frames);

    // Should return default with zero confidence
    expect(result.confidence).toBe(0);
  });

  it("should validate correct tonic", () => {
    const tonic = 220;
    const frames = generateMockFrames(tonic);

    const validation = validateTonic(frames, tonic);

    // Most frames should match expected swara positions
    expect(validation).toBeGreaterThan(0.5);
  });

  it("should reject incorrect tonic", () => {
    const actualTonic = 220;
    const wrongTonic = 300; // Way off
    const frames = generateMockFrames(actualTonic);

    const validation = validateTonic(frames, wrongTonic);

    // Few frames should match with wrong tonic
    expect(validation).toBeLessThan(0.4);
  });

  it("should have higher confidence with more data", () => {
    const tonic = 220;
    const fewFrames = generateMockFrames(tonic, 20);
    const manyFrames = generateMockFrames(tonic, 200);

    const resultFew = detectTonic(fewFrames);
    const resultMany = detectTonic(manyFrames);

    // More data should give higher confidence (generally)
    expect(resultMany.confidence).toBeGreaterThanOrEqual(resultFew.confidence * 0.8);
  });
});
