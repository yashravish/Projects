'use client';

import { motion } from 'motion/react';
import { useRouter } from 'next/navigation';
import { ArrowRight } from 'lucide-react';

export function CTASection() {
  const router = useRouter();

  return (
    <section className="relative px-6 md:px-12 py-24 sm:py-32">
      {/* #1 — Proper centered container matching the card grid width */}
      <div className="max-w-3xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center gap-6"
        >
          <h2
            className="font-display text-3xl sm:text-5xl md:text-6xl text-center"
            style={{ color: 'var(--foreground)' }}
          >
            Ready to{' '}
            <span className="italic" style={{ color: 'var(--accent)' }}>
              visualize
            </span>
            ?
          </h2>

          <p className="mt-6 mx-auto max-w-md text-lg text-center leading-relaxed" style={{ color: '#c9c4b8', textWrap: 'balance' }}>
            Open the studio, drop your favorite track, and watch it become
            something you&apos;ve never seen before.
          </p>

          {/* #2 — Prominent CTA matching the hero button pattern */}
          <motion.button
            onClick={() => router.push('/studio')}
            className="cta-launch group inline-flex items-center gap-3 cursor-pointer"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            id="cta-bottom-launch"
          >
            Open Studio
            <ArrowRight
              size={18}
              strokeWidth={2}
              className="transition-transform duration-300 group-hover:translate-x-1"
            />
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
