import type { Metadata } from 'next';
import { ArchitectureExplainer } from '@/components/architecture/architecture-explainer';
import { DiagramSVG } from '@/components/architecture/diagram-svg';

export const metadata: Metadata = {
  title: 'Architecture — AI Music Canvas',
  description: 'How AI Music Canvas is built: App Router, Zustand, Web Audio API, Canvas 2D, and the engineering decisions behind it.',
};

export default function ArchitecturePage() {
  return (
    <div className="min-h-screen pt-20 pb-16 px-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-16 space-y-4">
          <p className="text-caption" style={{ color: 'var(--accent)' }}>
            Engineering Deep Dive
          </p>
          <h1 className="font-display text-4xl sm:text-6xl" style={{ color: 'var(--foreground)' }}>
            How it&apos;s <span className="italic" style={{ color: 'var(--accent)' }}>built</span>
          </h1>
          <p className="text-base max-w-2xl" style={{ color: 'rgba(255,255,255,0.5)' }}>
            AI Music Canvas is a Next.js application that transforms audio into real-time
            visual art using the Web Audio API and Canvas 2D. Here&apos;s how the pieces fit together.
          </p>
        </div>

        {/* Architecture Diagram */}
        <section className="mb-16">
          <h2 className="text-caption mb-6" style={{ color: 'var(--accent)' }}>
            System Architecture
          </h2>
          <div className="glass p-6 sm:p-8">
            <DiagramSVG />
          </div>
        </section>

        {/* Explainer sections */}
        <ArchitectureExplainer />
      </div>
    </div>
  );
}
