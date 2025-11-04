/**
 * End-to-end pipeline tests
 */

import { describe, it, expect } from "vitest";
import { transcribe } from "../src/pipeline/transcribe";
import { toMIDI, toJSON, fromJSON } from "../src/index";

describe("Transcription Pipeline", () => {
  /**
   * Generate a simple test signal with known swaras
   */
  function generateTestSignal(sampleRate: number, tonicHz: number): Float32Array {
    const duration = 2; // seconds
    const numSamples = Math.floor(duration * sampleRate);
    const samples = new Float32Array(numSamples);

    // Create a sequence: S (0.5s), R2 (0.5s), G3 (0.5s), P (0.5s)
    const swaraFreqs = [
      tonicHz, // S
      tonicHz * Math.pow(2, 204 / 1200), // R2
      tonicHz * Math.pow(2, 386 / 1200), // G3
      tonicHz * Math.pow(2, 702 / 1200), // P
    ];

    const samplesPerSwara = Math.floor(numSamples / 4);

    for (let i = 0; i < numSamples; i++) {
      const swaraIndex = Math.floor(i / samplesPerSwara);
      if (swaraIndex < swaraFreqs.length) {
        const freq = swaraFreqs[swaraIndex];
        samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate) * 0.5;
      }
    }

    return samples;
  }

  it("should transcribe a simple test signal", async () => {
    const sampleRate = 22050;
    const tonicHz = 220; // A3
    const signal = generateTestSignal(sampleRate, tonicHz);

    const result = await transcribe(signal, {
      sampleRate,
      detector: "yin",
      tonicHz: "auto",
    });

    // Check tonic detection
    expect(result.tonic.hz).toBeGreaterThan(210);
    expect(result.tonic.hz).toBeLessThan(230);

    // Should have detected some swaras
    expect(result.swaras.length).toBeGreaterThan(0);

    // Check that times are increasing
    for (let i = 1; i < result.swaras.length; i++) {
      expect(result.swaras[i].start).toBeGreaterThanOrEqual(result.swaras[i - 1].start);
    }

    // Check that swaras have valid properties
    for (const swara of result.swaras) {
      expect(swara.end).toBeGreaterThan(swara.start);
      expect(swara.confidence).toBeGreaterThan(0);
      expect(swara.confidence).toBeLessThanOrEqual(1);
      expect(swara.centsFromSa).toBeGreaterThanOrEqual(0);
      expect(swara.centsFromSa).toBeLessThan(1200);
    }
  });

  it("should detect expected swaras in sequence", async () => {
    const sampleRate = 22050;
    const tonicHz = 220;
    const signal = generateTestSignal(sampleRate, tonicHz);

    const result = await transcribe(signal, {
      sampleRate,
      detector: "yin",
      tonicHz,
      ragaHint: null,
    });

    // Should detect at least some of S, R2, G3, P
    const detectedSwaras = result.swaras.map((s) => s.swara);
    const expectedSwaras = ["S", "R2", "G3", "P"];

    // At least 2 of the expected swaras should be detected
    const matches = expectedSwaras.filter((s) => detectedSwaras.includes(s));
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });

  it("should respect raga hint", async () => {
    const sampleRate = 22050;
    const signal = generateTestSignal(sampleRate, 220);

    const result = await transcribe(signal, {
      sampleRate,
      detector: "yin",
      tonicHz: 220,
      ragaHint: "Mohanam",
    });

    // All detected swaras should be in Mohanam (S, R2, G3, P, D2)
    const mohanamSwaras = ["S", "R2", "G3", "P", "D2"];

    for (const swara of result.swaras) {
      expect(mohanamSwaras).toContain(swara.swara);
    }
  });

  it("should handle silent audio", async () => {
    const signal = new Float32Array(22050); // 1 second of silence

    const result = await transcribe(signal, {
      sampleRate: 22050,
      detector: "yin",
    });

    // Should complete without errors
    expect(result.swaras.length).toBe(0);
    expect(result.notes.length).toBe(0);
  });

  it("should export to MIDI without errors", async () => {
    const signal = generateTestSignal(22050, 220);

    const transcription = await transcribe(signal, {
      sampleRate: 22050,
      detector: "yin",
      tonicHz: 220,
    });

    const midiBytes = toMIDI(transcription);

    // Should produce valid MIDI file
    expect(midiBytes).toBeInstanceOf(Uint8Array);
    expect(midiBytes.length).toBeGreaterThan(0);

    // Check MIDI header
    expect(midiBytes[0]).toBe(0x4d); // 'M'
    expect(midiBytes[1]).toBe(0x54); // 'T'
    expect(midiBytes[2]).toBe(0x68); // 'h'
    expect(midiBytes[3]).toBe(0x64); // 'd'
  });

  it("should export to and import from JSON", async () => {
    const signal = generateTestSignal(22050, 220);

    const transcription = await transcribe(signal, {
      sampleRate: 22050,
      detector: "yin",
      tonicHz: 220,
    });

    // Export to JSON
    const json = toJSON(transcription);
    expect(json).toBeTruthy();

    // Import from JSON
    const imported = fromJSON(json);

    // Should match original
    expect(imported.tonic.hz).toBeCloseTo(transcription.tonic.hz, 2);
    expect(imported.swaras.length).toBe(transcription.swaras.length);
    expect(imported.notes.length).toBe(transcription.notes.length);
  });

  it("should run faster than 3x real-time", async () => {
    const sampleRate = 22050;
    const duration = 10; // seconds
    const signal = new Float32Array(duration * sampleRate);

    // Generate 10 seconds of test audio
    for (let i = 0; i < signal.length; i++) {
      signal[i] = Math.sin((2 * Math.PI * 220 * i) / sampleRate) * 0.5;
    }

    const startTime = Date.now();

    await transcribe(signal, {
      sampleRate,
      detector: "yin",
    });

    const elapsed = (Date.now() - startTime) / 1000; // seconds

    // Should process 10s audio in less than 3.3 seconds (3x real-time)
    expect(elapsed).toBeLessThan(3.3);
  });
});
