/**
 * Decode audio file data into an AudioBuffer.
 * Compatible with React Query (returns a Promise).
 */

import { getAudioContext } from './audio-context';

export interface DecodeResult {
  buffer: AudioBuffer;
  duration: number;
  sampleRate: number;
  numberOfChannels: number;
}

/** Decode an audio file into an AudioBuffer. */
export async function decodeAudioFile(file: File): Promise<DecodeResult> {
  const ctx = getAudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const buffer = await ctx.decodeAudioData(arrayBuffer);

  return {
    buffer,
    duration: buffer.duration,
    sampleRate: buffer.sampleRate,
    numberOfChannels: buffer.numberOfChannels,
  };
}

/**
 * Extract downsampled waveform peaks from an AudioBuffer.
 * Used for timeline visualization — no need for full-resolution data.
 */
export function extractWaveformPeaks(
  buffer: AudioBuffer,
  targetSamples: number = 1000
): Float32Array {
  const channelData = buffer.getChannelData(0);
  const blockSize = Math.floor(channelData.length / targetSamples);
  const peaks = new Float32Array(targetSamples);

  for (let i = 0; i < targetSamples; i++) {
    const start = i * blockSize;
    let max = 0;
    for (let j = start; j < start + blockSize && j < channelData.length; j++) {
      const abs = Math.abs(channelData[j]);
      if (abs > max) max = abs;
    }
    peaks[i] = max;
  }

  return peaks;
}
