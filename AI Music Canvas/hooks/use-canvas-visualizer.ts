'use client';

import { useEffect, useRef } from 'react';
import { renderer } from '@/lib/canvas/renderer';
import { useStudioStore } from '@/store/studio-store';
import type { FrequencyData } from '@/types/audio';

export function useCanvasVisualizer(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  getFrequencyData: (() => FrequencyData | null) | null,
  reducedMotion: boolean
) {
  const resizeTimer = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const container = canvas.parentElement;
    if (!container) return;

    // Start renderer
    renderer.start(canvas, {
      getFrequencyData: getFrequencyData ?? null,
      getState: () => {
        const state = useStudioStore.getState();
        return {
          mode: state.mode,
          controls: state.controls,
          hasAudio: state.audioBuffer !== null && state.playback.isPlaying,
          reducedMotion,
        };
      },
    });

    // Resize observer (debounced at ~150ms)
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(resizeTimer.current);
      resizeTimer.current = requestAnimationFrame(() => {
        if (container) {
          renderer.resize(container.clientWidth, container.clientHeight);
        }
      });
    });
    observer.observe(container);

    return () => {
      renderer.stop();
      observer.disconnect();
      cancelAnimationFrame(resizeTimer.current);
    };
  }, [canvasRef, getFrequencyData, reducedMotion]);
}
