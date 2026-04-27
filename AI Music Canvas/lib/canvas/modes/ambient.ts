/**
 * Ambient mode: cool, ethereal, floating.
 * Tiny high-count particles with ribbon trails, gradient mesh background.
 */

import { ParticleSystem } from '../particles';
import type { FrequencyData } from '@/types/audio';
import type { ControlValues } from '@/types/studio';

const COLORS = ['#6FA8DC', '#B19CD9', '#E8E6F0', '#8BB8E8', '#9AADD6'];

let particleSystem: ParticleSystem | null = null;
let meshPhase = 0;

function drawGradientMesh(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  energy: number,
  bgMotion: number,
  time: number
): void {
  const motion = bgMotion / 100;
  meshPhase += 0.005 * motion;

  // Breathing gradient mesh
  const breathe = Math.sin(meshPhase) * 0.015 * motion;

  const cx1 = w * (0.3 + Math.sin(time * 0.0008) * 0.15 * motion);
  const cy1 = h * (0.4 + Math.cos(time * 0.0006) * 0.1 * motion);
  const cx2 = w * (0.7 + Math.sin(time * 0.001) * 0.1 * motion);
  const cy2 = h * (0.6 + Math.cos(time * 0.0007) * 0.12 * motion);

  // First gradient
  const g1 = ctx.createRadialGradient(cx1, cy1, 0, cx1, cy1, w * 0.5);
  g1.addColorStop(0, `rgba(111, 168, 220, ${0.06 + breathe + energy * 0.04})`);
  g1.addColorStop(1, 'transparent');
  ctx.fillStyle = g1;
  ctx.fillRect(0, 0, w, h);

  // Second gradient
  const g2 = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, w * 0.4);
  g2.addColorStop(0, `rgba(177, 156, 217, ${0.04 + breathe + energy * 0.03})`);
  g2.addColorStop(1, 'transparent');
  ctx.fillStyle = g2;
  ctx.fillRect(0, 0, w, h);
}

export function initAmbient(width: number, height: number, particleCount: number): void {
  particleSystem = new ParticleSystem(
    {
      count: particleCount,
      minSize: 1,
      maxSize: 4,
      colors: COLORS,
      shape: 'circle',
      speed: 0.15,
      gravity: -0.01,
      friction: 0.998,
      fadeIn: 0.2,
      fadeOut: 0.3,
    },
    width,
    height
  );
}

export function renderAmbient(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData | null,
  controls: ControlValues,
  time: number
): void {
  const energy = data ? data.energy : 0.05;
  const treble = data ? data.treble : 0.05;
  const intensity = controls.intensity / 100;

  // Dark cool background
  ctx.fillStyle = '#08090D';
  ctx.fillRect(0, 0, w, h);

  // Gradient mesh
  drawGradientMesh(ctx, w, h, energy * intensity, controls.backgroundMotion, time);

  // Particles
  if (!particleSystem) {
    initAmbient(w, h, controls.particleCount);
  }
  if (particleSystem) {
    particleSystem.resize(w, h);
    particleSystem.setCount(controls.particleCount);
    particleSystem.update(energy * intensity * 0.5);

    // Draw with extra glow for ambient feel
    ctx.save();
    if (controls.glow > 0) {
      ctx.shadowBlur = controls.glow * 0.5;
      ctx.shadowColor = 'rgba(111, 168, 220, 0.3)';
    }
    particleSystem.draw(ctx, controls.blur * 1.5, controls.glow * intensity);
    ctx.restore();
  }

  // Ribbon waveform (treble-reactive)
  if (data) {
    drawRibbonWaveform(ctx, w, h, data, controls, treble);
  }
}

function drawRibbonWaveform(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: FrequencyData,
  controls: ControlValues,
  treble: number
): void {
  const centerY = h * 0.5;
  const ribbonWidth = controls.waveformThickness * (1 + treble * 2);

  ctx.save();
  ctx.globalAlpha = 0.3 + treble * 0.3;

  const gradient = ctx.createLinearGradient(0, centerY - 50, 0, centerY + 50);
  gradient.addColorStop(0, 'rgba(111, 168, 220, 0.6)');
  gradient.addColorStop(0.5, 'rgba(177, 156, 217, 0.8)');
  gradient.addColorStop(1, 'rgba(111, 168, 220, 0.6)');

  ctx.strokeStyle = gradient;
  ctx.lineWidth = ribbonWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const step = Math.max(1, Math.floor(data.timeDomain.length / (w / 2)));
  ctx.beginPath();

  for (let i = 0; i < data.timeDomain.length; i += step) {
    const v = data.timeDomain[i] / 128.0;
    const y = centerY + (v - 1) * h * 0.2 * (controls.intensity / 100);
    const x = (i / data.timeDomain.length) * w;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();
  ctx.restore();
}

export function resizeAmbient(width: number, height: number): void {
  if (particleSystem) particleSystem.resize(width, height);
}

export function destroyAmbient(): void {
  particleSystem = null;
  meshPhase = 0;
}
