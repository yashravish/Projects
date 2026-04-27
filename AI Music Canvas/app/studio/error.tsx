'use client';

import { useEffect } from 'react';
import { RotateCcw } from 'lucide-react';

export default function StudioError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[AMC:Studio] Error boundary caught:', error);
  }, [error]);

  return (
    <div className="h-screen flex items-center justify-center p-8" style={{ background: '#0A0A0B' }}>
      <div className="glass max-w-md w-full p-8 text-center space-y-6">
        <div
          className="w-16 h-16 mx-auto rounded-full flex items-center justify-center"
          style={{ background: 'rgba(255, 45, 122, 0.1)', border: '1px solid rgba(255, 45, 122, 0.2)' }}
        >
          <span className="text-2xl">🎵</span>
        </div>

        <div className="space-y-2">
          <h2 className="font-display text-2xl" style={{ color: 'var(--foreground)' }}>
            Studio crashed
          </h2>
          <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
            {error.message || 'The canvas engine hit an unexpected error. This might be a browser compatibility issue.'}
          </p>
        </div>

        <button
          onClick={reset}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer"
          style={{
            background: 'rgba(var(--accent-rgb), 0.1)',
            border: '1px solid rgba(var(--accent-rgb), 0.2)',
            color: 'var(--accent)',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(var(--accent-rgb), 0.2)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(var(--accent-rgb), 0.1)'; }}
        >
          <RotateCcw size={16} strokeWidth={1.5} />
          Restart studio
        </button>
      </div>
    </div>
  );
}
