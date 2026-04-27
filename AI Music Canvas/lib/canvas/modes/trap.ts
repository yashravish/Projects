/**
 * Trap mode: hard, geometric, neon.
 * Sharp shapes, snappy beat pulses, grid lines, screen-shake on bass.
 */

import { ParticleSystem } from '../particles';
import type { FrequencyData } from '@/types/audio';
import type { ControlValues } from '@/types/studio';

const COLORS = ['#FF2D7A', '#00F0FF', '#FFFFFF', '#FF5C99', '#33F2FF'];

let particleSystem: ParticleSystem | null = null;
let shakeX = 0;
let shakeY = 0;

function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  bass: number,
  bgMotion: number
): void {
  const motion = bgMotion / 100;
  const gridSize = 60;
  const gridOpacity = 0.03 + bass * 0.08 * motion;

  ctx.save();
  ctx.strokeStyle = `rgba(255, 45, 122, ${gridOpacity})`;
  ctx.lineWidth = 1;

  // Vertical lines
  for (let x = 0; x <= w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  // Horizontal lines
  for (let y = 0; y <= h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Flash on bass
  if (bass > 0.7) {
    const flashOpacity = (bass - 0.7) * 1.5 * motion;
    ctx.strokeStyle = `rgba(0, 240, 255, ${flashOpacity * 0.3})`;
    ctx.lineWidth = 2;

    const flashY = h * (0.3 + Math.random() * 0.4);
    ctx.beginPath();
    ctx.moveTo(0, flashY);
    ctx.lineTo(w, flashY);
    ctx.stroke();
  }

  ctx.restore();
}

export function initTrap(width: number, height: number, particleCount: number): void {
  // Mix of squares and triangles
  particleSystem = new ParticleSystem(
    {
      count: particleCount,
      minSize: 2,
      maxSize: 12,
      colors: COLORS,
      shape: Math.random() > 0.5 ? 'square' : 'triangle',
      speed: 0.8,
      gravity: 0,
      friction: 0.96,
      fadeIn: 0.05,
      fadeOut: 0.1,
    },
    width,
    height
  );

  // Randomize shapes within system
  for (const p of particleSystem.particles) {
    p.shape = Math.random() > 0.4 ? 'square' : 'triangle';
  }
}

export function renderTrap(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData | null,
  controls: ControlValues,
  _time: number,
  reducedMotion: boolean = false
): void {
  const energy = data ? data.energy : 0.1;
  const bass = data ? data.bass : 0.1;
  const intensity = controls.intensity / 100;

  // Screen shake on bass (disabled for reduced motion)
  if (!reducedMotion && bass > 0.75) {
    const shakeAmount = (bass - 0.75) * 8 * intensity;
    shakeX = (Math.random() - 0.5) * shakeAmount;
    shakeY = (Math.random() - 0.5) * shakeAmount;
  } else {
    shakeX *= 0.9;
    shakeY *= 0.9;
  }

  ctx.save();
  ctx.translate(shakeX, shakeY);

  // Near-black background
  ctx.fillStyle = '#050507';
  ctx.fillRect(-10, -10, w + 20, h + 20);

  // Grid
  drawGrid(ctx, w, h, bass * intensity, controls.backgroundMotion);

  // Particles
  if (!particleSystem) {
    initTrap(w, h, controls.particleCount);
  }
  if (particleSystem) {
    particleSystem.resize(w, h);
    particleSystem.setCount(controls.particleCount);

    // Snappy pulse on beats
    const pulseEnergy = bass > 0.5 ? energy * intensity * 2 : energy * intensity * 0.5;
    particleSystem.update(pulseEnergy);
    particleSystem.draw(ctx, controls.blur * 0.5, controls.glow * intensity);
  }

  // Sharp waveform
  if (data) {
    drawTrapWaveform(ctx, w, h, data, controls);
  }

  ctx.restore();
}

function drawTrapWaveform(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData,
  controls: ControlValues
): void {
  const centerY = h * 0.5;
  const sliceWidth = w / data.timeDomain.length;

  // Magenta top, cyan bottom
  ctx.save();

  for (let pass = 0; pass < 2; pass++) {
    ctx.beginPath();
    ctx.strokeStyle = pass === 0 ? 'rgba(255, 45, 122, 0.5)' : 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = controls.waveformThickness * (pass === 0 ? 1 : 0.7);
    ctx.lineCap = 'butt';

    const offset = pass === 0 ? 0 : 2;

    for (let i = 0; i < data.timeDomain.length; i++) {
      const v = data.timeDomain[i] / 128.0;
      const y = centerY + (v - 1) * h * 0.3 * (controls.intensity / 100) + offset;
      const x = i * sliceWidth;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.restore();
}

export function resizeTrap(width: number, height: number): void {
  if (particleSystem) particleSystem.resize(width, height);
}

export function destroyTrap(): void {
  particleSystem = null;
  shakeX = 0;
  shakeY = 0;
}
