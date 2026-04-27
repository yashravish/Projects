/**
 * Alchemist mode: warm, analog, grain.
 * Large soft-edged orbs, film grain, vinyl-ring concentric circles on snare hits.
 */

import { ParticleSystem } from '../particles';
import type { FrequencyData } from '@/types/audio';
import type { ControlValues } from '@/types/studio';

const COLORS = ['#E8B65A', '#8B6F47', '#F4E8D0', '#C4943D', '#D4A94E'];

let particleSystem: ParticleSystem | null = null;
let grainCanvas: HTMLCanvasElement | null = null;
let grainCtx: CanvasRenderingContext2D | null = null;

function ensureGrainCanvas(width: number, height: number): void {
  if (!grainCanvas) {
    grainCanvas = document.createElement('canvas');
    grainCtx = grainCanvas.getContext('2d');
  }
  if (grainCanvas.width !== width || grainCanvas.height !== height) {
    grainCanvas.width = width;
    grainCanvas.height = height;
  }
}

function drawGrain(ctx: CanvasRenderingContext2D, w: number, h: number, intensity: number): void {
  ensureGrainCanvas(w, h);
  if (!grainCtx || !grainCanvas) return;

  const imgData = grainCtx.createImageData(w, h);
  const data = imgData.data;
  const alpha = Math.floor(intensity * 12);

  for (let i = 0; i < data.length; i += 4) {
    const v = Math.random() * 255;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = alpha;
  }

  grainCtx.putImageData(imgData, 0, 0);
  ctx.drawImage(grainCanvas, 0, 0);
}

function drawVinylRings(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  bass: number
): void {
  if (bass < 0.6) return;

  const cx = w / 2;
  const cy = h / 2;
  const rings = 5;
  const maxRadius = Math.min(w, h) * 0.4;
  const ringOpacity = (bass - 0.6) * 2;

  ctx.save();
  ctx.strokeStyle = `rgba(232, 182, 90, ${ringOpacity * 0.15})`;
  ctx.lineWidth = 1;

  for (let i = 0; i < rings; i++) {
    const radius = (maxRadius / rings) * (i + 1) * (0.8 + bass * 0.4);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.restore();
}

export function initAlchemist(width: number, height: number, particleCount: number): void {
  particleSystem = new ParticleSystem(
    {
      count: Math.min(particleCount, 300),
      minSize: 8,
      maxSize: 35,
      colors: COLORS,
      shape: 'circle',
      speed: 0.3,
      gravity: 0,
      friction: 0.99,
      fadeIn: 0.15,
      fadeOut: 0.2,
    },
    width,
    height
  );
}

export function renderAlchemist(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData | null,
  controls: ControlValues,
  _time: number
): void {
  const energy = data ? data.energy : 0.1;
  const bass = data ? data.bass : 0.1;
  const intensity = controls.intensity / 100;

  // Background
  ctx.fillStyle = '#0A0A0B';
  ctx.fillRect(0, 0, w, h);

  // Warm radial glow
  const gradient = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, w * 0.5);
  gradient.addColorStop(0, `rgba(232, 182, 90, ${0.03 + energy * 0.04 * intensity})`);
  gradient.addColorStop(0.5, `rgba(139, 111, 71, ${0.02 + energy * 0.02 * intensity})`);
  gradient.addColorStop(1, 'transparent');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  // Vinyl rings on strong bass
  drawVinylRings(ctx, w, h, bass * intensity);

  // Particles
  if (!particleSystem) {
    initAlchemist(w, h, controls.particleCount);
  }
  if (particleSystem) {
    particleSystem.resize(w, h);
    particleSystem.setCount(Math.min(controls.particleCount, 300));
    particleSystem.update(energy * intensity);
    particleSystem.draw(ctx, controls.blur, controls.glow * intensity);
  }

  // Film grain
  drawGrain(ctx, w, h, intensity * 0.5);

  // Waveform
  if (data) {
    drawWaveform(ctx, w, h, data, controls);
  }
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData,
  controls: ControlValues
): void {
  const centerY = h * 0.5;
  const sliceWidth = w / data.timeDomain.length;

  ctx.beginPath();
  ctx.strokeStyle = `rgba(232, 182, 90, 0.4)`;
  ctx.lineWidth = controls.waveformThickness;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let i = 0; i < data.timeDomain.length; i++) {
    const v = data.timeDomain[i] / 128.0;
    const y = centerY + (v - 1) * h * 0.25 * (controls.intensity / 100);
    const x = i * sliceWidth;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();
}

export function resizeAlchemist(width: number, height: number): void {
  if (particleSystem) particleSystem.resize(width, height);
}

export function destroyAlchemist(): void {
  particleSystem = null;
  grainCanvas = null;
  grainCtx = null;
}
