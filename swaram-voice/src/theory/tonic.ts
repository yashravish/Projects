/**
 * Tonic (Sa) detection for Carnatic music
 */

import type { Hz, PitchFrame, TonicEstimate } from "../types";

/**
 * Detect tonic from pitch frames using histogram method
 */
export function detectTonic(frames: PitchFrame[]): TonicEstimate {
  // Extract all voiced f0 values
  const f0Values = frames.filter((f) => f.f0 !== null && f.voicing > 0.5).map((f) => f.f0!);

  if (f0Values.length === 0) {
    return { hz: 220, confidence: 0 }; // Default A3
  }

  // Convert to log frequency (semitones from arbitrary reference)
  const referenceHz = 55; // A1
  const logF0s = f0Values.map((f0) => 12 * Math.log2(f0 / referenceHz));

  // Build histogram in semitone bins
  const minLog = Math.min(...logF0s);
  const maxLog = Math.max(...logF0s);
  const numBins = 120; // 10 octaves
  const binSize = (maxLog - minLog) / numBins;

  const histogram = new Float32Array(numBins);

  for (const logF0 of logF0s) {
    const bin = Math.floor((logF0 - minLog) / binSize);
    if (bin >= 0 && bin < numBins) {
      histogram[bin]++;
    }
  }

  // Smooth histogram with Gaussian kernel
  const smoothed = gaussianSmooth(histogram, 2.0);

  // Find peaks and prefer the lowest strong peak to avoid octave-up bias
  let maxCount = 0;
  for (let i = 0; i < smoothed.length; i++) {
    if (smoothed[i] > maxCount) maxCount = smoothed[i];
  }

  const threshold = maxCount * 0.4;
  const candidateBins: number[] = [];
  for (let i = 1; i < smoothed.length - 1; i++) {
    const isPeak = smoothed[i] >= smoothed[i - 1] && smoothed[i] >= smoothed[i + 1];
    if (isPeak && smoothed[i] >= threshold) {
      candidateBins.push(i);
    }
  }

  const chosenBin = candidateBins.length > 0 ? Math.min(...candidateBins) : smoothed.indexOf(maxCount as unknown as number);

  // Convert chosen bin back to Hz
  const logTonic = minLog + chosenBin * binSize;
  const tonicHz = referenceHz * Math.pow(2, logTonic / 12);

  // Refine using mean-shift around peak
  let refinedTonic = meanShiftRefine(f0Values, tonicHz, 0.05); // 5% tolerance

  // Fold octave errors by nudging towards the median of voiced f0s
  const sorted = [...f0Values].sort((a, b) => a - b);
  const medianF0 = sorted.length % 2 === 1
    ? sorted[(sorted.length - 1) / 2]
    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;

  if (medianF0 > 0) {
    // Bring refined tonic within ~[0.7x, 1.5x] of the median by octave folding
    while (refinedTonic > medianF0 * 1.5) {
      refinedTonic /= 2;
    }
    while (refinedTonic < medianF0 * 0.7) {
      refinedTonic *= 2;
    }
  }

  // Prefer the lowest stable pitch present in the material as Sa
  if (sorted.length > 0) {
    const minF0 = sorted[0];
    if (refinedTonic > minF0 * 1.1) {
      refinedTonic = minF0;
    }
  }

  // Calculate confidence based on peak sharpness
  const peakHeight = smoothed[chosenBin];
  const avgHeight = smoothed.reduce((sum, val) => sum + val, 0) / smoothed.length;
  const confidence = Math.min(1, (peakHeight - avgHeight) / avgHeight);

  return {
    hz: refinedTonic,
    confidence: Math.max(0, Math.min(1, confidence)),
  };
}

/**
 * Refine tonic estimate using mean-shift
 */
function meanShiftRefine(f0Values: Hz[], initialTonic: Hz, tolerance: number): Hz {
  let tonic = initialTonic;
  const maxIterations = 10;

  for (let iter = 0; iter < maxIterations; iter++) {
    const lowerBound = tonic * (1 - tolerance);
    const upperBound = tonic * (1 + tolerance);

    // Find f0 values near current tonic
    const nearby = f0Values.filter((f0) => f0 >= lowerBound && f0 <= upperBound);

    if (nearby.length === 0) {
      break;
    }

    // Calculate mean
    const newTonic = nearby.reduce((sum, f0) => sum + f0, 0) / nearby.length;

    // Check convergence
    if (Math.abs(newTonic - tonic) < 0.5) {
      tonic = newTonic;
      break;
    }

    tonic = newTonic;
  }

  return tonic;
}

/**
 * Apply Gaussian smoothing to histogram
 */
function gaussianSmooth(data: Float32Array, sigma: number): Float32Array {
  const smoothed = new Float32Array(data.length);
  const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
  const halfKernel = Math.floor(kernelSize / 2);

  // Generate Gaussian kernel
  const kernel = new Float32Array(kernelSize);
  let sum = 0;

  for (let i = 0; i < kernelSize; i++) {
    const x = i - halfKernel;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }

  // Normalize kernel
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= sum;
  }

  // Apply convolution with edge clamping to avoid attenuating boundaries
  for (let i = 0; i < data.length; i++) {
    let value = 0;
    let weightSum = 0;

    for (let j = 0; j < kernelSize; j++) {
      let idx = i + j - halfKernel;
      if (idx < 0) idx = 0;
      if (idx >= data.length) idx = data.length - 1;
      value += data[idx] * kernel[j];
      weightSum += kernel[j];
    }

    smoothed[i] = weightSum > 0 ? value / weightSum : 0;
  }

  return smoothed;
}

/**
 * Validate tonic using pitch class profile
 */
export function validateTonic(frames: PitchFrame[], candidateTonic: Hz): number {
  // Convert f0 values to cents from candidate tonic
  const centsValues: number[] = [];

  for (const frame of frames) {
    if (frame.f0 !== null && frame.voicing > 0.5) {
      const cents = 1200 * Math.log2(frame.f0 / candidateTonic);
      // Wrap to 0-1200 cents (one octave)
      const wrappedCents = ((cents % 1200) + 1200) % 1200;
      centsValues.push(wrappedCents);
    }
  }

  if (centsValues.length === 0) {
    return 0;
  }

  // Expected peaks for Carnatic music (in cents from Sa)
  const expectedPeaks = [0, 204, 294, 386, 498, 702, 906, 1088];
  const tolerance = 35; // cents (tighter to reduce false positives)

  // Count how many cents values fall near expected peaks
  let matchCount = 0;

  for (const cents of centsValues) {
    for (const peak of expectedPeaks) {
      if (Math.abs(cents - peak) < tolerance) {
        matchCount++;
        break;
      }
    }
  }

  return matchCount / centsValues.length;
}
