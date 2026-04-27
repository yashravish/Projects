'use client';

import { useEffect, useState, useCallback } from 'react';
import { useStudioStore } from '@/store/studio-store';

interface KeyboardHandlers {
  onPlay: () => void;
  onSeekForward: () => void;
  onSeekBackward: () => void;
}

export function useKeyboardShortcuts(handlers: KeyboardHandlers) {
  const [showShortcuts, setShowShortcuts] = useState(false);
  const hasAudio = useStudioStore((s) => s.audioBuffer !== null);
  const setMode = useStudioStore((s) => s.setMode);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Ignore if focus is in an input/textarea
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (hasAudio) handlers.onPlay();
          break;

        case 'ArrowRight':
          e.preventDefault();
          if (hasAudio) handlers.onSeekForward();
          break;

        case 'ArrowLeft':
          e.preventDefault();
          if (hasAudio) handlers.onSeekBackward();
          break;

        case 'Digit1':
        case 'Numpad1':
          setMode('alchemist');
          break;

        case 'Digit2':
        case 'Numpad2':
          setMode('ambient');
          break;

        case 'Digit3':
        case 'Numpad3':
          setMode('trap');
          break;

        case 'Digit4':
        case 'Numpad4':
          setMode('orchestral');
          break;

        case 'Slash':
          if (e.shiftKey) {
            // ? key
            e.preventDefault();
            setShowShortcuts((prev) => !prev);
          }
          break;

        case 'Escape':
          setShowShortcuts(false);
          break;
      }
    },
    [hasAudio, handlers, setMode]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return { showShortcuts, setShowShortcuts };
}
