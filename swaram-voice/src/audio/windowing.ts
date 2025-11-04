/**
 * Windowing functions for signal processing
 */

/**
 * Generate a Hann window
 */
export function hann(length: number): Float32Array {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return window;
}

/**
 * Generate a Hamming window
 */
export function hamming(length: number): Float32Array {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return window;
}

/**
 * Generate a Blackman window
 */
export function blackman(length: number): Float32Array {
  const window = new Float32Array(length);
  const a0 = 0.42;
  const a1 = 0.5;
  const a2 = 0.08;

  for (let i = 0; i < length; i++) {
    const x = (2 * Math.PI * i) / (length - 1);
    window[i] = a0 - a1 * Math.cos(x) + a2 * Math.cos(2 * x);
  }
  return window;
}

/**
 * Apply a window function to a signal segment
 */
export function applyWindow(signal: Float32Array, window: Float32Array): Float32Array {
  if (signal.length !== window.length) {
    throw new Error("Signal and window must have the same length");
  }

  const windowed = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    windowed[i] = signal[i] * window[i];
  }
  return windowed;
}
