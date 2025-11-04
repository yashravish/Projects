/**
 * Audio resampling utilities
 */

/**
 * Resample audio to target sample rate using linear interpolation
 * Includes simple anti-aliasing for downsampling
 */
export function resample(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) {
    return samples;
  }

  // If downsampling, apply simple boxcar low-pass filter to reduce aliasing
  let processedSamples = samples;
  if (toRate < fromRate) {
    const ratio = fromRate / toRate;
    const filterLength = Math.ceil(ratio);
    processedSamples = applyBoxcarFilter(samples, filterLength);
  }

  const ratio = fromRate / toRate;
  const newLength = Math.floor(processedSamples.length / ratio);
  const resampled = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, processedSamples.length - 1);
    const frac = srcIndex - srcIndexFloor;

    // Linear interpolation
    resampled[i] = processedSamples[srcIndexFloor] * (1 - frac) + processedSamples[srcIndexCeil] * frac;
  }

  return resampled;
}

/**
 * Apply simple boxcar (moving average) filter
 */
function applyBoxcarFilter(samples: Float32Array, length: number): Float32Array {
  if (length <= 1) {
    return samples;
  }

  const filtered = new Float32Array(samples.length);
  const halfLength = Math.floor(length / 2);

  for (let i = 0; i < samples.length; i++) {
    let sum = 0;
    let count = 0;

    for (let j = -halfLength; j <= halfLength; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < samples.length) {
        sum += samples[idx];
        count++;
      }
    }

    filtered[i] = sum / count;
  }

  return filtered;
}

/**
 * Decimate (downsample) by an integer factor with anti-aliasing
 */
export function decimate(samples: Float32Array, factor: number): Float32Array {
  if (factor === 1) {
    return samples;
  }

  // Simple moving average as low-pass filter
  const filterLength = factor * 2;
  const filtered = new Float32Array(samples.length);

  for (let i = 0; i < samples.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = -filterLength; j <= filterLength; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < samples.length) {
        sum += samples[idx];
        count++;
      }
    }
    filtered[i] = sum / count;
  }

  // Downsample
  const newLength = Math.floor(samples.length / factor);
  const decimated = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    decimated[i] = filtered[i * factor];
  }

  return decimated;
}
