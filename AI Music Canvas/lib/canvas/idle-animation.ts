/**
 * Procedural ambient animation for no-audio state.
 * Shows the visualizer is alive without requiring AudioContext.
 */

import type { ControlValues } from '@/types/studio';
import type { StyleMode } from '@/types/studio';
import { renderAlchemist, initAlchemist, destroyAlchemist } from './modes/alchemist';
import { renderAmbient, initAmbient, destroyAmbient } from './modes/ambient';
import { renderTrap, initTrap, destroyTrap } from './modes/trap';
import { renderOrchestral, initOrchestral, destroyOrchestral } from './modes/orchestral';
import type { FrequencyData } from '@/types/audio';

let time = 0;

/**
 * Generate synthetic frequency data from Perlin-like noise.
 * Produces gentle, low-energy oscillations that preview the visual mode.
 */
function generateSyntheticData(): FrequencyData {
  const length = 1024;
  const frequency = new Uint8Array(length);
  const timeDomain = new Uint8Array(length);

  for (let i = 0; i < length; i++) {
    // Smooth sine-based frequency data
    const f = Math.sin(i * 0.02 + time * 0.01) * 30 +
              Math.sin(i * 0.05 + time * 0.008) * 20 +
              40;
    frequency[i] = Math.max(0, Math.min(255, f));

    // Gentle waveform
    const t = Math.sin(i * 0.01 + time * 0.005) * 15 + 128;
    timeDomain[i] = Math.max(0, Math.min(255, t));
  }

  return {
    frequency,
    timeDomain,
    bass: 0.15 + Math.sin(time * 0.008) * 0.05,
    mid: 0.1 + Math.sin(time * 0.012) * 0.04,
    treble: 0.08 + Math.sin(time * 0.015) * 0.03,
    energy: 0.12 + Math.sin(time * 0.01) * 0.04,
  };
}

export function renderIdleAnimation(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  mode: StyleMode,
  controls: ControlValues,
  reducedMotion: boolean
): void {
  time++;

  if (reducedMotion) {
    // Static: just clear with background
    ctx.fillStyle = '#0A0A0B';
    ctx.fillRect(0, 0, w, h);
    return;
  }

  const syntheticData = generateSyntheticData();

  switch (mode) {
    case 'alchemist':
      renderAlchemist(ctx, w, h, syntheticData, controls, time);
      break;
    case 'ambient':
      renderAmbient(ctx, w, h, syntheticData, controls, time);
      break;
    case 'trap':
      renderTrap(ctx, w, h, syntheticData, controls, time, false);
      break;
    case 'orchestral':
      renderOrchestral(ctx, w, h, syntheticData, controls, time);
      break;
  }
}

export function initIdleMode(mode: StyleMode, w: number, h: number, count: number): void {
  destroyAllModes();
  switch (mode) {
    case 'alchemist': initAlchemist(w, h, count); break;
    case 'ambient': initAmbient(w, h, count); break;
    case 'trap': initTrap(w, h, count); break;
    case 'orchestral': initOrchestral(w, h, count); break;
  }
}

function destroyAllModes(): void {
  destroyAlchemist();
  destroyAmbient();
  destroyTrap();
  destroyOrchestral();
}
