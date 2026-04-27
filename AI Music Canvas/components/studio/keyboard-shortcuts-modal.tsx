'use client';

import { motion, AnimatePresence } from 'motion/react';
import { X } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const shortcuts = [
  { key: 'Space', action: 'Play / Pause' },
  { key: '←', action: 'Seek back 5s' },
  { key: '→', action: 'Seek forward 5s' },
  { key: '1', action: 'Alchemist mode' },
  { key: '2', action: 'Ambient mode' },
  { key: '3', action: 'Trap mode' },
  { key: '4', action: 'Orchestral mode' },
  { key: 'R', action: 'Start/stop recording' },
  { key: '?', action: 'Toggle this panel' },
  { key: 'Esc', action: 'Close modals' },
];

export function KeyboardShortcutsModal({ isOpen, onClose }: KeyboardShortcutsModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-[100]"
            style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed top-1/2 left-1/2 z-[101] w-80 glass p-6"
            style={{ transform: 'translate(-50%, -50%)' }}
            initial={{ opacity: 0, scale: 0.95, y: '-48%', x: '-50%' }}
            animate={{ opacity: 1, scale: 1, y: '-50%', x: '-50%' }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
                Keyboard Shortcuts
              </h3>
              <button
                onClick={onClose}
                className="p-1 rounded transition-colors duration-150 cursor-pointer"
                style={{ color: 'rgba(255,255,255,0.3)' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--foreground)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.3)'; }}
                aria-label="Close shortcuts panel"
              >
                <X size={16} strokeWidth={1.5} />
              </button>
            </div>

            <div className="space-y-2">
              {shortcuts.map((s) => (
                <div key={s.key} className="flex items-center justify-between py-1">
                  <span className="text-xs" style={{ color: 'rgba(255,255,255,0.5)' }}>
                    {s.action}
                  </span>
                  <kbd
                    className="px-2 py-0.5 rounded text-[11px] font-mono"
                    style={{
                      background: 'rgba(255,255,255,0.06)',
                      border: '1px solid rgba(255,255,255,0.08)',
                      color: 'var(--accent)',
                    }}
                  >
                    {s.key}
                  </kbd>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
