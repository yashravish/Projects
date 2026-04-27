'use client';

import { motion } from 'motion/react';
import { Upload, Layers, Sliders, Download } from 'lucide-react';
import type { ReactNode } from 'react';

interface Feature {
  icon: ReactNode;
  title: string;
  subtitle: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: <Upload size={24} strokeWidth={1.75} />,
    title: 'Drop to Decode',
    subtitle: 'Instant audio analysis',
    description:
      'Drag any audio file onto the canvas. The Web Audio API decodes your track, extracts frequency data in real time, and auto-detects sections.',
  },
  {
    icon: <Layers size={24} strokeWidth={1.75} />,
    title: 'Beat-Reactive Visuals',
    subtitle: 'Every frame responds',
    description:
      'Particles, waveforms, and backgrounds react to bass kicks, treble hits, and mid-range energy. Your music drives every frame.',
  },
  {
    icon: <Sliders size={24} strokeWidth={1.75} />,
    title: 'Four Style Modes',
    subtitle: 'Distinct visual languages',
    description:
      'Alchemist, Ambient, Trap, Orchestral — each mode is its own visual world with unique color, motion, and texture.',
  },
  {
    icon: <Download size={24} strokeWidth={1.75} />,
    title: 'Export as Video',
    subtitle: 'Share your creation',
    description:
      'Record the canvas with synced audio via MediaRecorder. Choose 15s, 30s, or full track — download as WebM, ready to share.',
  },
];

function FeatureCard({ feature, index }: { feature: Feature; index: number }) {
  return (
    <motion.div
      className="feature-card items-center text-center"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* #6 — Larger icon tile with more contrast */}
      <div className="card-icon-tile">
        {feature.icon}
      </div>

      {/* #7 — Eyebrow with proper spacing */}
      <p className="card-eyebrow">
        {feature.subtitle}
      </p>

      <h3 className="card-heading">
        {feature.title}
      </h3>

      {/* #5 — Body copy at proper contrast */}
      <p className="card-body">
        {feature.description}
      </p>
    </motion.div>
  );
}

export function FeatureShowcase() {
  return (
    <section className="relative px-6 md:px-12 py-24 sm:py-32">
      <div className="max-w-6xl mx-auto">
        <motion.div
          className="text-center mb-16 sm:mb-20"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <p className="hero-eyebrow" style={{ marginBottom: '16px' }}>
            How it works
          </p>
          {/* #8 — Changed from "From waveform to art" to avoid mirroring the hero */}
          <h2
            className="font-display text-3xl sm:text-5xl section-heading"
            style={{ color: 'var(--foreground)' }}
          >
            How sound becomes{' '}
            <span className="italic" style={{ color: 'var(--accent)' }}>visible</span>
          </h2>
        </motion.div>

        {/* #4 — Equal-height card grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 auto-rows-fr">
          {features.map((feature, i) => (
            <FeatureCard key={feature.title} feature={feature} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
