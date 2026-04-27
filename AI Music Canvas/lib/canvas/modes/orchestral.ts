/**
 * Orchestral mode: cinematic, sweeping, gold.
 * Medium particles with motion blur, sweeping arcs, cinematic vignette.
 */

import { ParticleSystem } from '../particles';
import type { FrequencyData } from '@/types/audio';
import type { ControlValues } from '@/types/studio';

const COLORS = ['#D4AF37', '#5C1A1B', '#FFFAF0', '#B8962E', '#E6C65C'];

let particleSystem: ParticleSystem | null = null;

function drawCinematicVignette(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  energy: number
): void {
  const gradient = ctx.createRadialGradient(
    w * 0.5, h * 0.5, w * 0.2,
    w * 0.5, h * 0.5, w * 0.7
  );
  gradient.addColorStop(0, 'transparent');
  gradient.addColorStop(0.6, `rgba(0, 0, 0, ${0.1 + (1 - energy) * 0.2})`);
  gradient.addColorStop(1, `rgba(0, 0, 0, ${0.4 + (1 - energy) * 0.2})`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);
}

function drawSweepingArcs(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData,
  controls: ControlValues,
  time: number
): void {
  const intensity = controls.intensity / 100;
  const cx = w * 0.5;
  const cy = h * 0.5;
  const maxRadius = Math.min(w, h) * 0.35;

  ctx.save();

  // Draw arcs that follow frequency peaks
  const arcCount = 8;
  for (let i = 0; i < arcCount; i++) {
    const freqIndex = Math.floor((i / arcCount) * data.frequency.length * 0.3);
    const freqValue = data.frequency[freqIndex] / 255;

    if (freqValue < 0.1) continue;

    const startAngle = (i / arcCount) * Math.PI * 2 + time * 0.0005;
    const sweepAngle = freqValue * Math.PI * 0.8 * intensity;
    const radius = maxRadius * (0.3 + freqValue * 0.7);
    const lineWidth = 1 + freqValue * 3 * intensity;

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sweepAngle);
    ctx.strokeStyle = `rgba(212, 175, 55, ${freqValue * 0.4 * intensity})`;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  ctx.restore();
}

export function initOrchestral(width: number, height: number, particleCount: number): void {
  particleSystem = new ParticleSystem(
    {
      count: Math.min(particleCount, 800),
      minSize: 2,
      maxSize: 8,
      colors: COLORS,
      shape: 'circle',
      speed: 0.5,
      gravity: -0.005,
      friction: 0.995,
      fadeIn: 0.15,
      fadeOut: 0.25,
    },
    width,
    height
  );
}

export function renderOrchestral(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData | null,
  controls: ControlValues,
  time: number
): void {
  const energy = data ? data.energy : 0.1;
  const mid = data ? data.mid : 0.1;
  const intensity = controls.intensity / 100;

  // Warm dark background
  ctx.fillStyle = '#0B0908';
  ctx.fillRect(0, 0, w, h);

  // Warm radial glow (swells on crescendos via mid energy)
  const gradient = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, w * 0.5);
  gradient.addColorStop(0, `rgba(212, 175, 55, ${0.03 + mid * 0.06 * intensity})`);
  gradient.addColorStop(0.4, `rgba(92, 26, 27, ${0.02 + mid * 0.03 * intensity})`);
  gradient.addColorStop(1, 'transparent');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  // Sweeping arcs
  if (data) {
    drawSweepingArcs(ctx, w, h, data, controls, time);
  }

  // Particles with motion blur
  if (!particleSystem) {
    initOrchestral(w, h, controls.particleCount);
  }
  if (particleSystem) {
    particleSystem.resize(w, h);
    particleSystem.setCount(Math.min(controls.particleCount, 800));

    // Swell on crescendos
    const swellEnergy = energy * intensity * (1 + mid * 0.5);
    particleSystem.update(swellEnergy);

    // Motion blur effect: semi-transparent clear
    ctx.save();
    ctx.globalAlpha = 0.8;
    particleSystem.draw(ctx, controls.blur * 1.2, controls.glow * intensity);
    ctx.restore();
  }

  // Cinematic vignette
  drawCinematicVignette(ctx, w, h, energy);

  // Waveform
  if (data) {
    drawOrchestralWaveform(ctx, w, h, data, controls);
  }
}

function drawOrchestralWaveform(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData,
  controls: ControlValues
): void {
  const centerY = h * 0.5;

  ctx.save();
  ctx.beginPath();

  const gradient = ctx.createLinearGradient(0, 0, w, 0);
  gradient.addColorStop(0, 'rgba(212, 175, 55, 0.1)');
  gradient.addColorStop(0.5, 'rgba(212, 175, 55, 0.5)');
  gradient.addColorStop(1, 'rgba(212, 175, 55, 0.1)');

  ctx.strokeStyle = gradient;
  ctx.lineWidth = controls.waveformThickness;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const step = Math.max(1, Math.floor(data.timeDomain.length / (w / 2)));

  for (let i = 0; i < data.timeDomain.length; i += step) {
    const v = data.timeDomain[i] / 128.0;
    const y = centerY + (v - 1) * h * 0.25 * (controls.intensity / 100);
    const x = (i / data.timeDomain.length) * w;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();
  ctx.restore();
}

export function resizeOrchestral(width: number, height: number): void {
  if (particleSystem) particleSystem.resize(width, height);
}

export function destroyOrchestral(): void {
  particleSystem = null;
}
