'use client';

import { useStudioStore } from '@/store/studio-store';
import { X } from 'lucide-react';
import { useEffect, useState } from 'react';

function ToastItem({ id, message, variant, duration = 4000 }: {
  id: string;
  message: string;
  variant: 'success' | 'error' | 'info';
  duration?: number;
}) {
  const removeToast = useStudioStore((s) => s.removeToast);
  const [progress, setProgress] = useState(100);
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    if (duration <= 0) return;
    const start = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100);
      setProgress(remaining);
      if (remaining <= 0) {
        clearInterval(interval);
        setIsExiting(true);
        setTimeout(() => removeToast(id), 300);
      }
    }, 50);
    return () => clearInterval(interval);
  }, [duration, id, removeToast]);

  const borderColor =
    variant === 'error' ? 'rgba(255,45,122,0.4)' :
    variant === 'success' ? 'rgba(111,168,220,0.4)' :
    'rgba(232,182,90,0.4)';

  const barColor =
    variant === 'error' ? '#FF2D7A' :
    variant === 'success' ? '#6FA8DC' :
    '#E8B65A';

  return (
    <div
      className="relative overflow-hidden rounded-lg"
      style={{
        background: 'rgba(15,15,17,0.95)',
        border: `1px solid ${borderColor}`,
        backdropFilter: 'blur(12px)',
        opacity: isExiting ? 0 : 1,
        transform: isExiting ? 'translateX(100%)' : 'translateX(0)',
        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        animation: 'slide-in-right 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      <div className="flex items-start gap-3 px-4 py-3">
        <p className="text-sm flex-1" style={{ color: 'rgba(255,255,255,0.85)' }}>
          {message}
        </p>
        <button
          onClick={() => {
            setIsExiting(true);
            setTimeout(() => removeToast(id), 300);
          }}
          className="flex-shrink-0 p-0.5 rounded transition-colors cursor-pointer"
          style={{ color: 'rgba(255,255,255,0.3)' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.7)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.3)'; }}
          aria-label="Dismiss notification"
        >
          <X size={14} strokeWidth={1.5} />
        </button>
      </div>
      {duration > 0 && (
        <div className="h-0.5 w-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
          <div
            className="h-full transition-none"
            style={{
              width: `${progress}%`,
              background: barColor,
            }}
          />
        </div>
      )}
    </div>
  );
}

export function ToastContainer() {
  const toasts = useStudioStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div
      className="fixed bottom-6 right-6 z-[10000] flex flex-col gap-2 w-80"
      role="region"
      aria-label="Notifications"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} {...toast} />
      ))}
    </div>
  );
}
