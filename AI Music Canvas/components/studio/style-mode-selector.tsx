'use client';

import { motion } from 'motion/react';
import { useStudioStore } from '@/store/studio-store';
import type { StyleMode } from '@/types/studio';

const MODES: { id: StyleMode; label: string; color: string }[] = [
  { id: 'alchemist', label: 'Alchemist', color: '#E8B65A' },
  { id: 'ambient', label: 'Ambient', color: '#6FA8DC' },
  { id: 'trap', label: 'Trap', color: '#FF2D7A' },
  { id: 'orchestral', label: 'Orchestral', color: '#D4AF37' },
];

export function StyleModeSelector() {
  const mode = useStudioStore((s) => s.mode);
  const setMode = useStudioStore((s) => s.setMode);

  return (
    <div className="space-y-2">
      <p className="text-caption">Style Mode</p>
      <div
        className="relative flex rounded-[var(--radius-button)] p-0.5"
        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className="relative flex-1 py-2 text-xs font-medium z-10 transition-colors duration-200 cursor-pointer rounded-[6px]"
            style={{
              color: mode === m.id ? '#0A0A0B' : 'rgba(255,255,255,0.5)',
            }}
            aria-pressed={mode === m.id}
            aria-label={`${m.label} visualization mode`}
          >
            {mode === m.id && (
              <motion.div
                layoutId="mode-pill"
                className="absolute inset-0 rounded-[6px]"
                style={{ background: m.color }}
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
            <span className="relative z-10">{m.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
