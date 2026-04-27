'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion, useSpring, useMotionValue } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import { startLandingWaves } from '@/lib/canvas/landing-waves';

export function Hero() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isTouch, setIsTouch] = useState(false);

  // Magnetic CTA: spring-animated displacement
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const springX = useSpring(mouseX, { stiffness: 150, damping: 15 });
  const springY = useSpring(mouseY, { stiffness: 150, damping: 15 });

  useEffect(() => {
    const mqlMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const mqlTouch = window.matchMedia('(pointer: coarse)');
    setReducedMotion(mqlMotion.matches);
    setIsTouch(mqlTouch.matches);

    const handleMotionChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    const handleTouchChange = (e: MediaQueryListEvent) => setIsTouch(e.matches);
    mqlMotion.addEventListener('change', handleMotionChange);
    mqlTouch.addEventListener('change', handleTouchChange);

    return () => {
      mqlMotion.removeEventListener('change', handleMotionChange);
      mqlTouch.removeEventListener('change', handleTouchChange);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    return startLandingWaves(canvas, reducedMotion);
  }, [reducedMotion]);

  function handleCtaMouseMove(e: React.MouseEvent) {
    if (isTouch || reducedMotion) return;
    const cta = ctaRef.current;
    if (!cta) return;
    const rect = cta.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const dx = e.clientX - centerX;
    const dy = e.clientY - centerY;
    const maxDisplacement = 8;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const maxDist = Math.max(rect.width, rect.height);
    const factor = Math.min(dist / maxDist, 1);
    mouseX.set(dx * factor * (maxDisplacement / maxDist));
    mouseY.set(dy * factor * (maxDisplacement / maxDist));
  }

  function handleCtaMouseLeave() {
    mouseX.set(0);
    mouseY.set(0);
  }

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Canvas background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        aria-hidden="true"
      />

      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
        {/* P1 #5 — Eyebrow: bumped to 13px, tighter letter-spacing, weight 500 */}
        <motion.p
          className="hero-eyebrow"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          Audio-to-Visual Art Engine
        </motion.p>

        {/* P1 #4 — Headline hierarchy: equalized sizes, tighter leading */}
        <motion.h1
          className="font-display mb-6"
          style={{ lineHeight: 0.95 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <span
            className="hero-title-line-1 block"
            style={{ color: 'var(--foreground)' }}
          >
            Sound
          </span>
          <span
            className="hero-title-line-2 block font-display italic"
            style={{ color: 'var(--accent)' }}
          >
            becomes art
          </span>
        </motion.h1>

        {/* P0 #2 — Body paragraph: lifted to #c9c4b8 (warm off-white), weight 400, max-width 56ch */}
        <motion.p
          className="hero-description"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          Drop an audio file onto the canvas and watch your music transform into
          living visual art — reactive, expressive, exportable.
        </motion.p>

        {/* P0 #1 — CTA: larger padding/font, prominent shadow */}
        {/* P0 #3 — Focus-visible ring via CSS class */}
        {/* P2 #9 — Arrow icon: stroke-width 2 */}
        {/* P2 #10 — Secondary action link below CTA */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center gap-4"
        >
          <div
            ref={ctaRef}
            onMouseMove={handleCtaMouseMove}
            onMouseLeave={handleCtaMouseLeave}
            className="inline-block"
          >
            <motion.button
              onClick={() => router.push('/studio')}
              className="cta-launch group inline-flex items-center gap-3 cursor-pointer"
              style={{
                x: springX,
                y: springY,
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              id="cta-launch-canvas"
            >
              Launch Canvas
              <ArrowRight
                size={18}
                strokeWidth={2}
                className="transition-transform duration-300 group-hover:translate-x-1"
              />
            </motion.button>
          </div>

          {/* P2 #10 — Secondary escape-hatch link */}
          <a
            href="/architecture"
            className="hero-secondary-link"
            id="cta-see-architecture"
          >
            See how it works →
          </a>
        </motion.div>
      </div>

      {/* Bottom gradient fade */}
      <div
        className="absolute bottom-0 left-0 right-0 h-32"
        style={{ background: 'linear-gradient(to top, var(--background), transparent)' }}
      />
    </section>
  );
}
