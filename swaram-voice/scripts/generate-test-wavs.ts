/**
 * Generate simple test WAV files for examples
 * Run with: node --loader ts-node/esm generate-test-wavs.ts
 */

import { writeFileSync } from "fs";

/**
 * Create a simple 16-bit PCM WAV file
 */
function createWAV(
  samples: Float32Array,
  sampleRate: number,
  filename: string
): void {
  const numChannels = 1; // Mono
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = samples.length * 2; // 2 bytes per sample (16-bit)

  // Create buffer
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF chunk
  view.setUint8(0, 0x52); // 'R'
  view.setUint8(1, 0x49); // 'I'
  view.setUint8(2, 0x46); // 'F'
  view.setUint8(3, 0x46); // 'F'
  view.setUint32(4, 36 + dataSize, true); // File size - 8
  view.setUint8(8, 0x57); // 'W'
  view.setUint8(9, 0x41); // 'A'
  view.setUint8(10, 0x56); // 'V'
  view.setUint8(11, 0x45); // 'E'

  // fmt chunk
  view.setUint8(12, 0x66); // 'f'
  view.setUint8(13, 0x6d); // 'm'
  view.setUint8(14, 0x74); // 't'
  view.setUint8(15, 0x20); // ' '
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // Audio format (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  view.setUint8(36, 0x64); // 'd'
  view.setUint8(37, 0x61); // 'a'
  view.setUint8(38, 0x74); // 't'
  view.setUint8(39, 0x61); // 'a'
  view.setUint32(40, dataSize, true);

  // Write samples
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const intSample = Math.floor(sample * 32767);
    view.setInt16(44 + i * 2, intSample, true);
  }

  writeFileSync(filename, Buffer.from(buffer));
  console.log(`Created ${filename}`);
}

/**
 * Generate a simple Sa-Re-Ga sequence
 */
function generateSaReGa(): Float32Array {
  const sampleRate = 22050;
  const duration = 2; // 2 seconds
  const samples = new Float32Array(duration * sampleRate);

  const tonic = 220; // A3
  const swaraFreqs = [
    tonic, // S
    tonic * Math.pow(2, 204 / 1200), // R2
    tonic * Math.pow(2, 386 / 1200), // G3
  ];

  const samplesPerSwara = Math.floor(samples.length / 3);

  for (let i = 0; i < samples.length; i++) {
    const swaraIndex = Math.floor(i / samplesPerSwara);
    if (swaraIndex < swaraFreqs.length) {
      const freq = swaraFreqs[swaraIndex];
      samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate) * 0.5;

      // Fade in/out at boundaries
      const posInSwara = i % samplesPerSwara;
      const fadeLength = sampleRate * 0.05; // 50ms fade

      if (posInSwara < fadeLength) {
        samples[i] *= posInSwara / fadeLength;
      } else if (posInSwara > samplesPerSwara - fadeLength) {
        samples[i] *= (samplesPerSwara - posInSwara) / fadeLength;
      }
    }
  }

  return samples;
}

/**
 * Generate a sustained Sa
 */
function generateSustainedSa(): Float32Array {
  const sampleRate = 22050;
  const duration = 1; // 1 second
  const samples = new Float32Array(duration * sampleRate);

  const tonic = 220; // A3

  for (let i = 0; i < samples.length; i++) {
    samples[i] = Math.sin((2 * Math.PI * tonic * i) / sampleRate) * 0.5;
  }

  return samples;
}

// Generate test files
createWAV(generateSaReGa(), 22050, "tests/fixtures/sa-re-ga.wav");
createWAV(generateSustainedSa(), 22050, "tests/fixtures/sustain-sa.wav");

console.log("\nTest WAV files created successfully!");
console.log("Run: npm run examples/node-file.ts");
