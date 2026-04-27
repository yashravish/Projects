'use client';

import { useCallback, useRef } from 'react';
import { useStudioStore } from '@/store/studio-store';

interface WaveformTimelineProps {
  onSeek: (time: number) => void;
}

export function WaveformTimeline({ onSeek }: WaveformTimelineProps) {
  const peaks = useStudioStore((s) => s.waveformPeaks);
  const duration = useStudioStore((s) => s.playback.duration);
  const currentTime = useStudioStore((s) => s.playback.currentTime);
  const sections = useStudioStore((s) => s.sections);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const progress = duration > 0 ? currentTime / duration : 0;

  const getTimeFromEvent = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      if (!containerRef.current || duration <= 0) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      return (x / rect.width) * duration;
    },
    [duration]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      isDragging.current = true;
      const time = getTimeFromEvent(e);
      onSeek(time);

      const handleMouseMove = (ev: MouseEvent) => {
        if (!isDragging.current) return;
        const t = getTimeFromEvent(ev);
        onSeek(t);
      };

      const handleMouseUp = () => {
        isDragging.current = false;
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };

      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    },
    [getTimeFromEvent, onSeek]
  );

  if (!peaks) return null;

  return (
    <div className="space-y-1.5">
      <p className="text-caption">Timeline</p>
      <div
        ref={containerRef}
        className="relative h-16 rounded-[var(--radius-button)] overflow-hidden"
        style={{
          background: 'rgba(255,255,255,0.02)',
          border: '1px solid rgba(255,255,255,0.04)',
          cursor: isDragging.current ? 'grabbing' : 'grab',
        }}
        onMouseDown={handleMouseDown}
        role="slider"
        aria-label="Audio timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        tabIndex={0}
      >
        {/* Section markers background */}
        {sections.map((section) => {
          const left = duration > 0 ? (section.startTime / duration) * 100 : 0;
          const width = duration > 0 ? ((section.endTime - section.startTime) / duration) * 100 : 0;
          return (
            <div
              key={section.id}
              className="absolute top-0 h-full"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: `${section.color}08`,
                borderRight: `1px solid ${section.color}20`,
              }}
            >
              <span
                className="absolute top-1 left-1 text-[9px] uppercase tracking-wider font-medium"
                style={{ color: `${section.color}80` }}
              >
                {section.label}
              </span>
            </div>
          );
        })}

        {/* Waveform */}
        <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
          {Array.from(peaks).map((peak, i) => {
            const x = (i / peaks.length) * 100;
            const barHeight = peak * 80;
            const isPast = (i / peaks.length) <= progress;
            return (
              <rect
                key={i}
                x={`${x}%`}
                y={`${50 - barHeight / 2}%`}
                width={`${Math.max(100 / peaks.length - 0.2, 0.1)}%`}
                height={`${Math.max(barHeight, 1)}%`}
                fill={isPast ? 'var(--accent)' : 'rgba(255,255,255,0.15)'}
                opacity={isPast ? 0.8 : 0.5}
                rx="0.5"
              />
            );
          })}
        </svg>

        {/* Playhead */}
        <div
          className="absolute top-0 w-0.5 h-full"
          style={{
            left: `${progress * 100}%`,
            background: 'var(--accent)',
            boxShadow: '0 0 8px rgba(var(--accent-rgb), 0.5)',
            transition: isDragging.current ? 'none' : 'left 0.1s linear',
          }}
        />
      </div>
    </div>
  );
}
