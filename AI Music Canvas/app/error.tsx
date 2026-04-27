'use client';

import { useEffect } from 'react';
import { RotateCcw } from 'lucide-react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[AMC] Root error boundary caught:', error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <div className="glass max-w-md w-full p-8 text-center space-y-6">
        <div className="w-16 h-16 mx-auto rounded-full flex items-center justify-center"
             style={{ background: 'rgba(255, 45, 122, 0.1)', border: '1px solid rgba(255, 45, 122, 0.2)' }}>
          <span className="text-2xl">⚠</span>
        </div>

        <div className="space-y-2">
          <h2 className="font-display text-2xl" style={{ color: 'var(--foreground)' }}>
            Something broke
          </h2>
          <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
            {error.message || 'An unexpected error occurred. The canvas engine encountered a problem.'}
          </p>
        </div>

        <button
          onClick={reset}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200"
          style={{
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.1)',
            color: 'var(--foreground)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255,255,255,0.06)';
          }}
        >
          <RotateCcw size={16} strokeWidth={1.5} />
          Try again
        </button>
      </div>
    </div>
  );
}
