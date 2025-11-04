/**
 * Unified audio source interface for Node and Browser
 */

import type { AudioSource, AudioSourceInput } from "../types";

/**
 * Convert various audio input formats to a uniform AudioSource
 */
export async function createAudioSource(
  input: Float32Array | AudioBuffer | ArrayBuffer | Buffer | AudioSourceInput
): Promise<AudioSource> {
  // Case 1: AudioSourceInput with explicit sample rate
  if (isAudioSourceInput(input)) {
    return {
      sampleRate: input.sampleRate,
      samples: input.samples,
      duration: input.samples.length / input.sampleRate,
    };
  }

  // Case 2: Already Float32Array (mono PCM) - default to 44100
  if (input instanceof Float32Array) {
    return {
      sampleRate: 44100, // Default assumption; caller should use AudioSourceInput for explicit SR
      samples: input,
      duration: input.length / 44100,
    };
  }

  // Case 3: Web Audio API AudioBuffer
  if (typeof AudioBuffer !== "undefined" && input instanceof AudioBuffer) {
    return fromAudioBuffer(input);
  }

  // Case 4: ArrayBuffer or Node Buffer (assume WAV format)
  const buffer =
    input instanceof ArrayBuffer
      ? input
      : (input as Buffer).buffer.slice(
          (input as Buffer).byteOffset,
          (input as Buffer).byteOffset + (input as Buffer).byteLength
        );
  return fromWAV(buffer);
}

/**
 * Type guard for AudioSourceInput
 */
function isAudioSourceInput(input: any): input is AudioSourceInput {
  return (
    input &&
    typeof input === "object" &&
    input.samples instanceof Float32Array &&
    typeof input.sampleRate === "number"
  );
}

/**
 * Extract mono audio from AudioBuffer
 */
function fromAudioBuffer(buffer: AudioBuffer): AudioSource {
  let samples: Float32Array;

  if (buffer.numberOfChannels === 1) {
    samples = buffer.getChannelData(0);
  } else {
    // Mix to mono by averaging channels
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    samples = new Float32Array(left.length);
    for (let i = 0; i < left.length; i++) {
      samples[i] = (left[i] + right[i]) / 2;
    }
  }

  return {
    sampleRate: buffer.sampleRate,
    samples,
    duration: buffer.duration,
  };
}

/**
 * Simple WAV decoder (supports PCM 16-bit and 32-bit float)
 */
function fromWAV(arrayBuffer: ArrayBufferLike): AudioSource {
  const view = new DataView(arrayBuffer);

  // Verify RIFF header
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== "RIFF") {
    throw new Error("Invalid WAV file: missing RIFF header");
  }

  // Verify WAVE format
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (wave !== "WAVE") {
    throw new Error("Invalid WAV file: missing WAVE format");
  }

  // Find fmt chunk
  let offset = 12;
  let fmtChunk = findChunk(view, offset, "fmt ");
  if (!fmtChunk) {
    throw new Error("Invalid WAV file: missing fmt chunk");
  }

  const audioFormat = view.getUint16(fmtChunk.offset, true);
  const numChannels = view.getUint16(fmtChunk.offset + 2, true);
  const sampleRate = view.getUint32(fmtChunk.offset + 4, true);
  const bitsPerSample = view.getUint16(fmtChunk.offset + 14, true);

  // Find data chunk
  const dataChunk = findChunk(view, 12, "data");
  if (!dataChunk) {
    throw new Error("Invalid WAV file: missing data chunk");
  }

  // Decode samples based on format
  let samples: Float32Array;

  if (audioFormat === 1) {
    // PCM
    if (bitsPerSample === 16) {
      samples = decodePCM16(view, dataChunk.offset, dataChunk.size, numChannels);
    } else if (bitsPerSample === 24) {
      throw new Error(
        `24-bit PCM is not supported. Please convert to 16-bit or 32-bit float WAV format.`
      );
    } else {
      throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}. Supported: 16-bit PCM, 32-bit float.`);
    }
  } else if (audioFormat === 3) {
    // IEEE Float
    samples = decodeFloat32(view, dataChunk.offset, dataChunk.size, numChannels);
  } else {
    throw new Error(
      `Unsupported audio format code: ${audioFormat}. Supported: PCM (1) and IEEE Float (3).`
    );
  }

  return {
    sampleRate,
    samples,
    duration: samples.length / sampleRate,
  };
}

/**
 * Find a chunk in a WAV file, tolerating non-audio chunks (JUNK, LIST, bext, iXML, etc.)
 */
function findChunk(view: DataView, startOffset: number, chunkId: string): { offset: number; size: number } | null {
  let offset = startOffset;
  const endOffset = view.byteLength;

  while (offset < endOffset - 8) {
    const id = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3)
    );
    const size = view.getUint32(offset + 4, true);

    if (id === chunkId) {
      return { offset: offset + 8, size };
    }

    // Skip this chunk and move to next
    offset += 8 + size;
    if (size % 2 === 1) offset++; // Skip padding byte for word alignment
  }

  return null;
}

/**
 * Decode 16-bit PCM to Float32Array (mono)
 */
function decodePCM16(view: DataView, offset: number, size: number, numChannels: number): Float32Array {
  const numSamples = size / 2 / numChannels;
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = view.getInt16(offset + (i * numChannels + ch) * 2, true);
      sum += sample / 32768.0;
    }
    samples[i] = sum / numChannels;
  }

  return samples;
}

/**
 * Decode 32-bit float to Float32Array (mono)
 */
function decodeFloat32(view: DataView, offset: number, size: number, numChannels: number): Float32Array {
  const numSamples = size / 4 / numChannels;
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      sum += view.getFloat32(offset + (i * numChannels + ch) * 4, true);
    }
    samples[i] = sum / numChannels;
  }

  return samples;
}
