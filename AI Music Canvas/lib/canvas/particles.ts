/**
 * Reusable particle system.
 * Configurable shape, size, color, physics — shared across all modes.
 */

export type ParticleShape = 'circle' | 'square' | 'triangle' | 'ring';

export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  baseSize: number;
  color: string;
  opacity: number;
  life: number;
  maxLife: number;
  shape: ParticleShape;
  rotation: number;
  rotationSpeed: number;
}

export interface ParticleConfig {
  count: number;
  minSize: number;
  maxSize: number;
  colors: string[];
  shape: ParticleShape;
  speed: number;
  gravity: number;
  friction: number;
  fadeIn: number;
  fadeOut: number;
}

export function createParticle(
  config: ParticleConfig,
  width: number,
  height: number
): Particle {
  const size = config.minSize + Math.random() * (config.maxSize - config.minSize);
  return {
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * config.speed,
    vy: (Math.random() - 0.5) * config.speed,
    size,
    baseSize: size,
    color: config.colors[Math.floor(Math.random() * config.colors.length)],
    opacity: 0,
    life: 0,
    maxLife: 200 + Math.random() * 300,
    shape: config.shape,
    rotation: Math.random() * Math.PI * 2,
    rotationSpeed: (Math.random() - 0.5) * 0.02,
  };
}

export function updateParticle(
  p: Particle,
  config: ParticleConfig,
  width: number,
  height: number,
  energy: number
): void {
  p.life++;
  p.x += p.vx + energy * (Math.random() - 0.5) * 2;
  p.y += p.vy + config.gravity;
  p.vx *= config.friction;
  p.vy *= config.friction;
  p.rotation += p.rotationSpeed;
  p.size = p.baseSize * (0.8 + energy * 0.6);

  // Fade in/out based on life
  const lifeRatio = p.life / p.maxLife;
  if (lifeRatio < config.fadeIn) {
    p.opacity = lifeRatio / config.fadeIn;
  } else if (lifeRatio > (1 - config.fadeOut)) {
    p.opacity = (1 - lifeRatio) / config.fadeOut;
  } else {
    p.opacity = 1;
  }

  // Wrap around
  if (p.x < -p.size) p.x = width + p.size;
  if (p.x > width + p.size) p.x = -p.size;
  if (p.y < -p.size) p.y = height + p.size;
  if (p.y > height + p.size) p.y = -p.size;
}

export function drawParticle(
  ctx: CanvasRenderingContext2D,
  p: Particle,
  blur: number,
  glow: number
): void {
  ctx.save();
  ctx.globalAlpha = p.opacity;
  ctx.translate(p.x, p.y);
  ctx.rotate(p.rotation);

  if (glow > 0) {
    ctx.shadowColor = p.color;
    ctx.shadowBlur = glow * 0.4;
  }

  if (blur > 0) {
    ctx.filter = `blur(${blur * 0.3}px)`;
  }

  ctx.fillStyle = p.color;

  switch (p.shape) {
    case 'circle':
      ctx.beginPath();
      ctx.arc(0, 0, p.size, 0, Math.PI * 2);
      ctx.fill();
      break;

    case 'square':
      ctx.fillRect(-p.size, -p.size, p.size * 2, p.size * 2);
      break;

    case 'triangle':
      ctx.beginPath();
      ctx.moveTo(0, -p.size);
      ctx.lineTo(-p.size * 0.866, p.size * 0.5);
      ctx.lineTo(p.size * 0.866, p.size * 0.5);
      ctx.closePath();
      ctx.fill();
      break;

    case 'ring':
      ctx.beginPath();
      ctx.arc(0, 0, p.size, 0, Math.PI * 2);
      ctx.strokeStyle = p.color;
      ctx.lineWidth = Math.max(1, p.size * 0.2);
      ctx.stroke();
      break;
  }

  ctx.restore();
}

/** Manage a pool of particles — create, update, draw, recycle. */
export class ParticleSystem {
  particles: Particle[] = [];
  config: ParticleConfig;
  width: number;
  height: number;

  constructor(config: ParticleConfig, width: number, height: number) {
    this.config = config;
    this.width = width;
    this.height = height;
    this.init();
  }

  init(): void {
    this.particles = [];
    const count = Math.min(this.config.count, 2000);
    for (let i = 0; i < count; i++) {
      const p = createParticle(this.config, this.width, this.height);
      p.life = Math.random() * p.maxLife;
      this.particles.push(p);
    }
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  setCount(count: number): void {
    const target = Math.min(count, 2000);
    if (target > this.particles.length) {
      for (let i = this.particles.length; i < target; i++) {
        this.particles.push(createParticle(this.config, this.width, this.height));
      }
    } else if (target < this.particles.length) {
      this.particles.length = target;
    }
    this.config.count = target;
  }

  update(energy: number): void {
    for (const p of this.particles) {
      updateParticle(p, this.config, this.width, this.height, energy);
      if (p.life >= p.maxLife) {
        // Recycle
        const fresh = createParticle(this.config, this.width, this.height);
        Object.assign(p, fresh);
      }
    }
  }

  draw(ctx: CanvasRenderingContext2D, blur: number, glow: number): void {
    for (const p of this.particles) {
      drawParticle(ctx, p, blur, glow);
    }
  }
}
