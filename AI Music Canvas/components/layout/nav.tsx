'use client';

import Link from 'next/link';
import { AudioWaveform, ArrowRight } from 'lucide-react';

export function Nav() {
  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
      style={{
        background: 'rgba(10,10,11,0.8)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}
    >
      <Link
        href="/"
        className="flex items-center gap-2.5 group transition-opacity duration-200 hover:opacity-80"
      >
        <AudioWaveform size={20} strokeWidth={1.5} style={{ color: 'var(--accent)' }} />
        <span className="text-sm font-medium tracking-tight" style={{ color: 'var(--foreground)' }}>
          AI Music Canvas
        </span>
      </Link>

      {/* #3 — Differentiated nav: "Architecture" is secondary, "Open Studio" is the primary action */}
      <div className="flex items-center gap-6">
        <Link href="/architecture" className="nav-link">
          Architecture
        </Link>
        <Link href="/studio" className="nav-cta">
          Open Studio
          <ArrowRight size={14} strokeWidth={2} />
        </Link>
      </div>
    </nav>
  );
}
