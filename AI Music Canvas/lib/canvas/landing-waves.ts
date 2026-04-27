/** Procedural golden wave animation for the landing hero canvas. */

interface WaveConfig {
  amplitude: number;
  frequency: number;
  speed: number;
  phase: number;
  opacity: number;
  lineWidth: number;
}

const WAVES: WaveConfig[] = [
  { amplitude: 40, frequency: 0.008, speed: 0.015, phase: 0, opacity: 0.25, lineWidth: 1.5 },
  { amplitude: 55, frequency: 0.006, speed: 0.012, phase: 2, opacity: 0.18, lineWidth: 1 },
  { amplitude: 30, frequency: 0.012, speed: 0.02, phase: 4, opacity: 0.3, lineWidth: 2 },
  { amplitude: 65, frequency: 0.004, speed: 0.008, phase: 1, opacity: 0.12, lineWidth: 0.8 },
  { amplitude: 20, frequency: 0.015, speed: 0.025, phase: 3, opacity: 0.15, lineWidth: 0.6 },
];

export function startLandingWaves(
  canvas: HTMLCanvasElement,
  reducedMotion: boolean
): () => void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};

  let animationId = 0;
  let time = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx!.scale(dpr, dpr);
  }

  resize();

  const resizeObserver = new ResizeObserver(() => {
    resize();
  });
  if (canvas.parentElement) {
    resizeObserver.observe(canvas.parentElement);
  }

  function drawWave(wave: WaveConfig, w: number, h: number, t: number) {
    if (!ctx) return;
    const centerY = h * 0.55;

    ctx.beginPath();
    ctx.strokeStyle = `rgba(232, 182, 90, ${wave.opacity})`;
    ctx.lineWidth = wave.lineWidth;

    for (let x = 0; x <= w; x += 2) {
      const y = centerY +
        Math.sin(x * wave.frequency + t * wave.speed + wave.phase) * wave.amplitude +
        Math.sin(x * wave.frequency * 1.5 + t * wave.speed * 0.7) * wave.amplitude * 0.3;

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  function render() {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Radial gradient background glow
    const gradient = ctx.createRadialGradient(w * 0.5, h * 0.55, 0, w * 0.5, h * 0.55, w * 0.6);
    gradient.addColorStop(0, 'rgba(232, 182, 90, 0.04)');
    gradient.addColorStop(1, 'transparent');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    for (const wave of WAVES) {
      drawWave(wave, w, h, time);
    }

    time += 1;

    if (!reducedMotion) {
      animationId = requestAnimationFrame(render);
    }
  }

  if (reducedMotion) {
    // Draw one static frame
    render();
  } else {
    animationId = requestAnimationFrame(render);
  }

  return () => {
    cancelAnimationFrame(animationId);
    resizeObserver.disconnect();
  };
}
