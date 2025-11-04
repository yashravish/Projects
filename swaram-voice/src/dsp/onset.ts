/**
 * Onset detection for note segmentation and tempo estimation
 */

/**
 * Simple energy-based onset detection
 */
export function detectOnsets(
  signal: Float32Array,
  sampleRate: number,
  hopSize: number,
  threshold: number = 0.3
): number[] {
  const onsets: number[] = [];
  const frameSize = hopSize * 2;
  const numFrames = Math.floor((signal.length - frameSize) / hopSize);

  // Calculate energy for each frame
  const energies = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const end = start + frameSize;
    let energy = 0;

    for (let j = start; j < end && j < signal.length; j++) {
      energy += signal[j] * signal[j];
    }

    energies[i] = Math.sqrt(energy / frameSize);
  }

  // Compute spectral flux (energy difference)
  const flux = new Float32Array(numFrames - 1);
  for (let i = 1; i < numFrames; i++) {
    const diff = energies[i] - energies[i - 1];
    flux[i - 1] = Math.max(0, diff); // Half-wave rectification
  }

  // Adaptive threshold
  const mean = flux.reduce((sum, val) => sum + val, 0) / flux.length;
  const adaptiveThreshold = mean * threshold;

  // Find peaks above threshold
  for (let i = 1; i < flux.length - 1; i++) {
    if (flux[i] > adaptiveThreshold && flux[i] > flux[i - 1] && flux[i] > flux[i + 1]) {
      const timeSeconds = (i * hopSize) / sampleRate;
      onsets.push(timeSeconds);
    }
  }

  return onsets;
}

/**
 * Estimate tempo from onset times using autocorrelation
 */
export function estimateTempo(onsets: number[], minBPM: number = 60, maxBPM: number = 180): number | null {
  if (onsets.length < 4) {
    return null; // Not enough onsets
  }

  // Calculate inter-onset intervals (IOIs)
  const iois: number[] = [];
  for (let i = 1; i < onsets.length; i++) {
    iois.push(onsets[i] - onsets[i - 1]);
  }

  // Convert BPM range to interval range (in seconds)
  const minInterval = 60 / maxBPM;
  const maxInterval = 60 / minBPM;

  // Build histogram of IOIs
  const numBins = 100;
  const histogram = new Float32Array(numBins);
  const binSize = (maxInterval - minInterval) / numBins;

  for (const ioi of iois) {
    if (ioi >= minInterval && ioi <= maxInterval) {
      const bin = Math.floor((ioi - minInterval) / binSize);
      if (bin >= 0 && bin < numBins) {
        histogram[bin]++;
      }
    }
  }

  // Find peak in histogram
  let maxBin = 0;
  let maxCount = 0;

  for (let i = 0; i < numBins; i++) {
    if (histogram[i] > maxCount) {
      maxCount = histogram[i];
      maxBin = i;
    }
  }

  if (maxCount === 0) {
    return null;
  }

  // Convert bin to tempo
  const interval = minInterval + maxBin * binSize;
  const tempo = 60 / interval;

  return tempo;
}
