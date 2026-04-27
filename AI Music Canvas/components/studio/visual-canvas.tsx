'use client';

import { memo, useRef, useCallback, useState, useEffect } from 'react';
import { useCanvasVisualizer } from '@/hooks/use-canvas-visualizer';
import { useStudioStore } from '@/store/studio-store';
import { getCurrentSection } from '@/lib/audio/sections';
import type { FrequencyData } from '@/types/audio';

interface VisualCanvasProps {
  getFrequencyData: (() => FrequencyData | null) | null;
  reducedMotion: boolean;
  isTouch: boolean;
}

function VisualCanvasInner({ getFrequencyData, reducedMotion, isTouch }: VisualCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [cursorHidden, setCursorHidden] = useState(false);

  const mode = useStudioStore((s) => s.mode);
  const isPlaying = useStudioStore((s) => s.playback.isPlaying);
  const currentTime = useStudioStore((s) => s.playback.currentTime);
  const sections = useStudioStore((s) => s.sections);

  useCanvasVisualizer(canvasRef, getFrequencyData, reducedMotion);

  // Cursor inactivity timeout (2s) during playback
  const resetCursorTimer = useCallback(() => {
    if (isTouch || reducedMotion) return;
    setCursorHidden(false);

    if (cursorTimerRef.current) clearTimeout(cursorTimerRef.current);

    if (isPlaying) {
      cursorTimerRef.current = setTimeout(() => {
        setCursorHidden(true);
      }, 2000);
    }
  }, [isPlaying, isTouch, reducedMotion]);

  useEffect(() => {
    return () => {
      if (cursorTimerRef.current) clearTimeout(cursorTimerRef.current);
    };
  }, []);

  // Dynamic aria-label
  const currentSection = getCurrentSection(sections, currentTime);
  const sectionLabel = currentSection ? `${currentSection.label} section` : 'no section';
  const ariaLabel = `${mode} visualization, ${sectionLabel}`;

  return (
    <div
      className="relative w-full h-full flex-1"
      style={{
        cursor: cursorHidden ? 'none' : 'crosshair',
        minHeight: 300,
      }}
      onMouseMove={resetCursorTimer}
      onMouseEnter={resetCursorTimer}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        aria-label={ariaLabel}
        role="img"
      />
    </div>
  );
}

export const VisualCanvas = memo(VisualCanvasInner);
VisualCanvas.displayName = 'VisualCanvas';
