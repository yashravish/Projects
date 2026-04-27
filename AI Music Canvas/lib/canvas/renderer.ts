/**
 * Main render loop orchestrator.
 * OWNS the requestAnimationFrame loop. The React hook is a thin bridge.
 */

import type { FrequencyData } from '@/types/audio';
import type { StyleMode, ControlValues } from '@/types/studio';
import { renderAlchemist, initAlchemist, resizeAlchemist, destroyAlchemist } from './modes/alchemist';
import { renderAmbient, initAmbient, resizeAmbient, destroyAmbient } from './modes/ambient';
import { renderTrap, initTrap, resizeTrap, destroyTrap } from './modes/trap';
import { renderOrchestral, initOrchestral, resizeOrchestral, destroyOrchestral } from './modes/orchestral';
import { renderIdleAnimation, initIdleMode } from './idle-animation';

interface RendererRefs {
  getFrequencyData: (() => FrequencyData | null) | null;
  getState: () => { mode: StyleMode; controls: ControlValues; hasAudio: boolean; reducedMotion: boolean };
}

let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let animationId = 0;
let refs: RendererRefs | null = null;
let currentMode: StyleMode | null = null;
let width = 0;
let height = 0;
let time = 0;

// Crossfade state
let prevMode: StyleMode | null = null;
let crossfadeProgress = 1;
let crossfadeCanvas: HTMLCanvasElement | null = null;
let crossfadeCtx: CanvasRenderingContext2D | null = null;
const CROSSFADE_DURATION = 36; // ~600ms at 60fps

function initMode(mode: StyleMode, w: number, h: number, count: number): void {
  switch (mode) {
    case 'alchemist': initAlchemist(w, h, count); break;
    case 'ambient': initAmbient(w, h, count); break;
    case 'trap': initTrap(w, h, count); break;
    case 'orchestral': initOrchestral(w, h, count); break;
  }
}

function renderMode(
  renderCtx: CanvasRenderingContext2D,
  mode: StyleMode,
  w: number,
  h: number,
  data: FrequencyData | null,
  controls: ControlValues,
  t: number,
  reducedMotion: boolean
): void {
  switch (mode) {
    case 'alchemist': renderAlchemist(renderCtx, w, h, data, controls, t); break;
    case 'ambient': renderAmbient(renderCtx, w, h, data, controls, t); break;
    case 'trap': renderTrap(renderCtx, w, h, data, controls, t, reducedMotion); break;
    case 'orchestral': renderOrchestral(renderCtx, w, h, data, controls, t); break;
  }
}

function resizeMode(mode: StyleMode, w: number, h: number): void {
  switch (mode) {
    case 'alchemist': resizeAlchemist(w, h); break;
    case 'ambient': resizeAmbient(w, h); break;
    case 'trap': resizeTrap(w, h); break;
    case 'orchestral': resizeOrchestral(w, h); break;
  }
}

function destroyMode(mode: StyleMode): void {
  switch (mode) {
    case 'alchemist': destroyAlchemist(); break;
    case 'ambient': destroyAmbient(); break;
    case 'trap': destroyTrap(); break;
    case 'orchestral': destroyOrchestral(); break;
  }
}

function ensureCrossfadeCanvas(w: number, h: number): void {
  if (!crossfadeCanvas) {
    crossfadeCanvas = document.createElement('canvas');
    crossfadeCtx = crossfadeCanvas.getContext('2d');
  }
  if (crossfadeCanvas.width !== w || crossfadeCanvas.height !== h) {
    crossfadeCanvas.width = w;
    crossfadeCanvas.height = h;
  }
}

function renderFrame(): void {
  if (!ctx || !refs || !canvas) return;

  const state = refs.getState();
  const mode = state.mode;
  const controls = state.controls;
  const hasAudio = state.hasAudio;
  const reducedMotion = state.reducedMotion;

  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Handle mode switch with crossfade
  if (currentMode !== mode) {
    if (currentMode !== null) {
      prevMode = currentMode;
      crossfadeProgress = 0;
    }
    initMode(mode, width, height, controls.particleCount);
    currentMode = mode;
  }

  // Get frequency data if audio is playing
  const data = (hasAudio && refs.getFrequencyData) ? refs.getFrequencyData() : null;

  if (crossfadeProgress < 1 && prevMode !== null) {
    // Render both modes and blend
    ensureCrossfadeCanvas(canvas.width, canvas.height);
    if (crossfadeCtx) {
      crossfadeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // Old mode in offscreen
      renderMode(crossfadeCtx, prevMode, width, height, data, controls, time, reducedMotion);
    }

    // New mode on main canvas
    renderMode(ctx, mode, width, height, data, controls, time, reducedMotion);

    // Blend: draw old mode with fading opacity
    if (crossfadeCanvas) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1 - crossfadeProgress;
      ctx.drawImage(crossfadeCanvas, 0, 0);
      ctx.restore();
    }

    crossfadeProgress += 1 / CROSSFADE_DURATION;
    if (crossfadeProgress >= 1) {
      crossfadeProgress = 1;
      if (prevMode !== null) {
        destroyMode(prevMode);
        prevMode = null;
      }
    }
  } else if (hasAudio) {
    renderMode(ctx, mode, width, height, data, controls, time, reducedMotion);
  } else {
    renderIdleAnimation(ctx, width, height, mode, controls, reducedMotion);
  }

  time++;
  animationId = requestAnimationFrame(renderFrame);
}

export const renderer = {
  start(
    canvasEl: HTMLCanvasElement,
    rendererRefs: RendererRefs
  ): void {
    canvas = canvasEl;
    ctx = canvasEl.getContext('2d');
    refs = rendererRefs;

    if (!ctx) return;

    const state = refs.getState();
    currentMode = null;
    time = 0;

    this.resize(canvasEl.parentElement?.clientWidth ?? canvasEl.clientWidth,
                canvasEl.parentElement?.clientHeight ?? canvasEl.clientHeight);

    initIdleMode(state.mode, width, height, state.controls.particleCount);

    animationId = requestAnimationFrame(renderFrame);
  },

  stop(): void {
    cancelAnimationFrame(animationId);
    if (currentMode) destroyMode(currentMode);
    if (prevMode) destroyMode(prevMode);
    currentMode = null;
    prevMode = null;
    canvas = null;
    ctx = null;
    refs = null;
    crossfadeCanvas = null;
    crossfadeCtx = null;
  },

  resize(w: number, h: number): void {
    width = w;
    height = h;

    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    if (currentMode) resizeMode(currentMode, w, h);
  },

  setMode(mode: StyleMode): void {
    // Mode change is handled inside renderFrame via state check
    // This is a no-op — the state drives the switch
  },
};
