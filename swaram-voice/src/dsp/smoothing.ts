/**
 * Smoothing and filtering utilities for pitch tracks
 */

import type { Hz, PitchFrame } from "../types";

/**
 * Apply median filter to pitch track to reduce jitter
 */
export function medianFilter(frames: PitchFrame[], windowSize: number = 5): PitchFrame[] {
  if (windowSize % 2 === 0) {
    windowSize++; // Ensure odd window size
  }

  const halfWindow = Math.floor(windowSize / 2);
  const filtered: PitchFrame[] = [];

  for (let i = 0; i < frames.length; i++) {
    const window: (Hz | null)[] = [];

    // Collect window of f0 values
    for (let j = -halfWindow; j <= halfWindow; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < frames.length && frames[idx].f0 !== null) {
        window.push(frames[idx].f0);
      }
    }

    let f0: Hz | null = null;
    if (window.length > 0) {
      // Calculate median
      window.sort((a, b) => a! - b!);
      const mid = Math.floor(window.length / 2);
      f0 = window.length % 2 === 0 ? (window[mid - 1]! + window[mid]!) / 2 : window[mid]!;
    }

    filtered.push({
      t: frames[i].t,
      f0,
      voicing: frames[i].voicing,
    });
  }

  return filtered;
}

/**
 * Apply moving average filter to smooth pitch track
 */
export function movingAverage(frames: PitchFrame[], windowSize: number = 3): PitchFrame[] {
  if (windowSize < 1) {
    return frames;
  }

  const halfWindow = Math.floor(windowSize / 2);
  const smoothed: PitchFrame[] = [];

  for (let i = 0; i < frames.length; i++) {
    let sum = 0;
    let count = 0;

    for (let j = -halfWindow; j <= halfWindow; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < frames.length && frames[idx].f0 !== null) {
        sum += frames[idx].f0!;
        count++;
      }
    }

    const f0 = count > 0 ? sum / count : null;

    smoothed.push({
      t: frames[i].t,
      f0,
      voicing: frames[i].voicing,
    });
  }

  return smoothed;
}

/**
 * Fill short gaps in pitch track (interpolation)
 */
export function fillGaps(frames: PitchFrame[], maxGapMs: number = 100, sampleRate: number = 22050): PitchFrame[] {
  const filled = [...frames];
  const maxGapFrames = Math.floor((maxGapMs / 1000) * sampleRate);

  for (let i = 0; i < filled.length; i++) {
    if (filled[i].f0 === null) {
      // Find previous voiced frame
      let prevIdx = i - 1;
      while (prevIdx >= 0 && filled[prevIdx].f0 === null) {
        prevIdx--;
      }

      // Find next voiced frame
      let nextIdx = i + 1;
      while (nextIdx < filled.length && filled[nextIdx].f0 === null) {
        nextIdx++;
      }

      // Interpolate if gap is small enough
      if (
        prevIdx >= 0 &&
        nextIdx < filled.length &&
        nextIdx - prevIdx <= maxGapFrames
      ) {
        const prevF0 = filled[prevIdx].f0!;
        const nextF0 = filled[nextIdx].f0!;
        const gapSize = nextIdx - prevIdx;

        for (let j = prevIdx + 1; j < nextIdx; j++) {
          const t = (j - prevIdx) / gapSize;
          filled[j].f0 = prevF0 * (1 - t) + nextF0 * t;
          filled[j].voicing = Math.max(filled[j].voicing, 0.5);
        }
      }
    }
  }

  return filled;
}

/**
 * Remove outliers using z-score method
 */
export function removeOutliers(frames: PitchFrame[], zThreshold: number = 3): PitchFrame[] {
  // Calculate mean and std dev of f0 values
  const f0Values = frames.filter((f) => f.f0 !== null).map((f) => f.f0!);

  if (f0Values.length === 0) {
    return frames;
  }

  const mean = f0Values.reduce((sum, val) => sum + val, 0) / f0Values.length;
  const variance =
    f0Values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / f0Values.length;
  const stdDev = Math.sqrt(variance);

  // Filter outliers
  return frames.map((frame) => {
    if (frame.f0 !== null) {
      const zScore = Math.abs((frame.f0 - mean) / stdDev);
      if (zScore > zThreshold) {
        return { ...frame, f0: null, voicing: 0 };
      }
    }
    return frame;
  });
}
