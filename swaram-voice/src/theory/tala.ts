/**
 * Tāla (rhythm/tempo) detection and grid generation
 */

import { detectOnsets, estimateTempo } from "../dsp/onset";

/**
 * Common tāla definitions
 */
export interface TalaDefinition {
  name: string;
  /** Number of beats (aksharas) per cycle (avartanam) */
  aksharas: number;
  /** Subdivision pattern (laghu, dhruta, anudruta) */
  pattern?: string;
}

export const COMMON_TALAS: Record<string, TalaDefinition> = {
  Adi: {
    name: "Adi",
    aksharas: 8,
    pattern: "4-2-2", // Chatusra Jati Triputa
  },
  Rupaka: {
    name: "Rupaka",
    aksharas: 6,
    pattern: "2-4",
  },
  Misra_Chapu: {
    name: "Misra Chapu",
    aksharas: 7,
    pattern: "3-2-2",
  },
  Khanda_Chapu: {
    name: "Khanda Chapu",
    aksharas: 5,
    pattern: "2-3",
  },
  Triputa: {
    name: "Triputa",
    aksharas: 7,
    pattern: "3-2-2",
  },
  Jhampa: {
    name: "Jhampa",
    aksharas: 10,
    pattern: "7-1-2",
  },
};

/**
 * Detect tempo from audio
 */
export function detectTempoFromAudio(
  signal: Float32Array,
  sampleRate: number,
  hopSize: number = 512
): number | null {
  // Detect onsets
  const onsets = detectOnsets(signal, sampleRate, hopSize, 0.3);

  if (onsets.length < 4) {
    return null;
  }

  // Estimate tempo from onsets
  return estimateTempo(onsets, 60, 180);
}

/**
 * Generate a beat grid for a given tempo and duration
 */
export function generateBeatGrid(
  tempo: number,
  durationSeconds: number,
  _tala?: TalaDefinition
): number[] {
  const beatInterval = 60 / tempo; // seconds per beat
  const numBeats = Math.floor(durationSeconds / beatInterval);
  
  const grid: number[] = [];

  for (let i = 0; i <= numBeats; i++) {
    grid.push(i * beatInterval);
  }

  return grid;
}

/**
 * Snap event times to beat grid
 */
export function snapToGrid(
  times: number[],
  beatGrid: number[],
  tolerance: number = 0.1
): number[] {
  return times.map((time) => {
    // Find nearest beat
    let nearestBeat = beatGrid[0];
    let minDistance = Math.abs(time - nearestBeat);

    for (const beat of beatGrid) {
      const distance = Math.abs(time - beat);
      if (distance < minDistance) {
        minDistance = distance;
        nearestBeat = beat;
      }
    }

    // Snap if within tolerance
    if (minDistance <= tolerance) {
      return nearestBeat;
    }

    return time;
  });
}

/**
 * Quantize event times to a beat grid
 */
export function quantizeToGrid(eventTimes: number[], tempo: number): number[] {
  const beatInterval = 60 / tempo;
  
  return eventTimes.map((time) => {
    const beatNumber = Math.round(time / beatInterval);
    return beatNumber * beatInterval;
  });
}

/**
 * Get tāla by name
 */
export function getTala(name: string): TalaDefinition | null {
  const normalized = name.toLowerCase().replace(/\s+/g, "_");
  
  for (const [key, tala] of Object.entries(COMMON_TALAS)) {
    if (key.toLowerCase().replace(/\s+/g, "_") === normalized) {
      return tala;
    }
  }

  return null;
}
