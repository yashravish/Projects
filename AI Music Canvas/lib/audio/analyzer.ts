/**
 * AnalyserNode setup and frequency/time-domain data extraction.
 * Provides band-separated energy values for the canvas renderer.
 */

import type { FrequencyData } from '@/types/audio';

export interface AnalyzerSetup {
  analyser: AnalyserNode;
  frequencyData: Uint8Array;
  timeDomainData: Uint8Array;
  getFrequencyData: () => FrequencyData;
  connectSource: (source: AudioNode) => void;
  disconnect: () => void;
}

/** Create an AnalyserNode and connect it in the audio graph. */
export function createAnalyzer(audioContext: AudioContext): AnalyzerSetup {
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.8;

  const bufferLength = analyser.frequencyBinCount;
  const frequencyData = new Uint8Array(bufferLength);
  const timeDomainData = new Uint8Array(bufferLength);

  // Band boundaries (approximate for music analysis)
  const bassEnd = Math.floor(bufferLength * 0.06);    // ~0-250Hz
  const midEnd = Math.floor(bufferLength * 0.25);      // ~250-2000Hz
  const trebleEnd = Math.floor(bufferLength * 0.7);     // ~2000-8000Hz

  function getFrequencyData(): FrequencyData {
    analyser.getByteFrequencyData(frequencyData);
    analyser.getByteTimeDomainData(timeDomainData);

    // Calculate band energies (0-1 range)
    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0;

    for (let i = 0; i < bassEnd; i++) {
      bassSum += frequencyData[i];
    }
    for (let i = bassEnd; i < midEnd; i++) {
      midSum += frequencyData[i];
    }
    for (let i = midEnd; i < trebleEnd; i++) {
      trebleSum += frequencyData[i];
    }
    for (let i = 0; i < bufferLength; i++) {
      totalSum += frequencyData[i];
    }

    const bass = bassEnd > 0 ? bassSum / (bassEnd * 255) : 0;
    const mid = (midEnd - bassEnd) > 0 ? midSum / ((midEnd - bassEnd) * 255) : 0;
    const treble = (trebleEnd - midEnd) > 0 ? trebleSum / ((trebleEnd - midEnd) * 255) : 0;
    const energy = bufferLength > 0 ? totalSum / (bufferLength * 255) : 0;

    return {
      frequency: frequencyData,
      timeDomain: timeDomainData,
      bass,
      mid,
      treble,
      energy,
    };
  }

  function connectSource(source: AudioNode) {
    source.connect(analyser);
    analyser.connect(audioContext.destination);
  }

  function disconnect() {
    try {
      analyser.disconnect();
    } catch {
      // Already disconnected
    }
  }

  return {
    analyser,
    frequencyData,
    timeDomainData,
    getFrequencyData,
    connectSource,
    disconnect,
  };
}
