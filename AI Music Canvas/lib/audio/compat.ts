/**
 * Browser compatibility layer for Web Audio API.
 * Handles AudioContext/webkitAudioContext and user-gesture requirements.
 */

type AudioContextConstructor = typeof AudioContext;

function getAudioContextClass(): AudioContextConstructor | null {
  if (typeof window === 'undefined') return null;

  const ctx = window.AudioContext ||
    (window as unknown as { webkitAudioContext?: AudioContextConstructor }).webkitAudioContext;

  return ctx || null;
}

/** Check if AudioContext is available in this browser. */
export function isAudioContextSupported(): boolean {
  return getAudioContextClass() !== null;
}

/** Create an AudioContext, using webkit fallback for older Safari. */
export function createAudioContext(options?: AudioContextOptions): AudioContext {
  const AudioCtx = getAudioContextClass();
  if (!AudioCtx) {
    throw new Error('AudioContext is not supported in this browser');
  }
  return new AudioCtx(options);
}

/**
 * Ensure AudioContext is in 'running' state.
 * Must be called inside a user gesture handler for Safari/iOS.
 */
export async function ensureAudioContextResumed(ctx: AudioContext): Promise<void> {
  if (ctx.state === 'suspended') {
    await ctx.resume();
  }
}

/** Detect MediaRecorder support and best available MIME type. */
export function detectMediaRecorderCompat(): {
  supported: boolean;
  mimeType: string | null;
} {
  if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined') {
    return { supported: false, mimeType: null };
  }

  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ];

  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return { supported: true, mimeType: mime };
    }
  }

  return { supported: false, mimeType: null };
}
