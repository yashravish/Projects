/**
 * Tests for pitch detection
 */

import { describe, it, expect } from "vitest";
import { YINDetector } from "../src/dsp/pitch/yin";
import { ACFDetector } from "../src/dsp/pitch/acf";

describe("Pitch Detection", () => {
  const sampleRate = 22050;

  /**
   * Generate a sine wave for testing
   */
  function generateSineWave(frequency: number, duration: number, sampleRate: number): Float32Array {
    const numSamples = Math.floor(duration * sampleRate);
    const samples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      samples[i] = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
    }

    return samples;
  }

  describe("YIN Detector", () => {
    it("should detect pitch of a pure 220 Hz tone (A3)", () => {
      const detector = new YINDetector();
      const signal = generateSineWave(220, 0.1, sampleRate);

      const result = detector.detect(signal, sampleRate);

      expect(result.f0).not.toBeNull();
      expect(result.f0!).toBeGreaterThan(215);
      expect(result.f0!).toBeLessThan(225);
      expect(result.voicing).toBeGreaterThan(0.7);
    });

    it("should detect pitch of a 440 Hz tone (A4)", () => {
      const detector = new YINDetector();
      const signal = generateSineWave(440, 0.1, sampleRate);

      const result = detector.detect(signal, sampleRate);

      expect(result.f0).not.toBeNull();
      expect(result.f0!).toBeGreaterThan(430);
      expect(result.f0!).toBeLessThan(450);
      expect(result.voicing).toBeGreaterThan(0.7);
    });

    it("should detect unvoiced for white noise", () => {
      const detector = new YINDetector();
      const signal = new Float32Array(2048);
      
      // Generate white noise
      for (let i = 0; i < signal.length; i++) {
        signal[i] = Math.random() * 2 - 1;
      }

      const result = detector.detect(signal, sampleRate);

      // Noise should have low voicing confidence
      expect(result.voicing).toBeLessThan(0.5);
    });

    it("should handle silent signal", () => {
      const detector = new YINDetector();
      const signal = new Float32Array(2048); // All zeros

      const result = detector.detect(signal, sampleRate);

      expect(result.f0).toBeNull();
      expect(result.voicing).toBe(0);
    });
  });

  describe("ACF Detector", () => {
    it("should detect pitch of a pure 220 Hz tone", () => {
      const detector = new ACFDetector();
      const signal = generateSineWave(220, 0.1, sampleRate);

      const result = detector.detect(signal, sampleRate);

      expect(result.f0).not.toBeNull();
      expect(result.f0!).toBeGreaterThan(210);
      expect(result.f0!).toBeLessThan(230);
    });

    it("should detect pitch of a 330 Hz tone (E4)", () => {
      const detector = new ACFDetector();
      const signal = generateSineWave(330, 0.1, sampleRate);

      const result = detector.detect(signal, sampleRate);

      expect(result.f0).not.toBeNull();
      expect(result.f0!).toBeGreaterThan(320);
      expect(result.f0!).toBeLessThan(340);
    });
  });

  describe("Detector Comparison", () => {
    it("YIN and ACF should produce similar results for pure tones", () => {
      const yinDetector = new YINDetector();
      const acfDetector = new ACFDetector();
      const signal = generateSineWave(260, 0.1, sampleRate);

      const yinResult = yinDetector.detect(signal, sampleRate);
      const acfResult = acfDetector.detect(signal, sampleRate);

      expect(yinResult.f0).not.toBeNull();
      expect(acfResult.f0).not.toBeNull();

      // Results should be within 5% of each other
      const diff = Math.abs(yinResult.f0! - acfResult.f0!);
      expect(diff).toBeLessThan(13); // 5% of 260 Hz
    });
  });
});
