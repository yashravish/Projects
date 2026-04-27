/**
 * Singleton AudioContext factory.
 * Lazy init — created on first user gesture to satisfy browser autoplay policy.
 */

import { createAudioContext, ensureAudioContextResumed } from './compat';

let instance: AudioContext | null = null;

/** Get or create the singleton AudioContext. Call inside a user gesture. */
export function getAudioContext(): AudioContext {
  if (!instance || instance.state === 'closed') {
    instance = createAudioContext();
  }
  return instance;
}

/** Resume the singleton AudioContext if suspended. Call on user interactions. */
export async function resumeAudioContext(): Promise<AudioContext> {
  const ctx = getAudioContext();
  await ensureAudioContextResumed(ctx);
  return ctx;
}

/** Close and discard the singleton AudioContext. */
export async function closeAudioContext(): Promise<void> {
  if (instance && instance.state !== 'closed') {
    await instance.close();
  }
  instance = null;
}
