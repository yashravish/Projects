'use client';

import { Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import { useStudioStore } from '@/store/studio-store';
import { formatTime } from '@/lib/utils/format';

interface TransportProps {
  onPlay: () => void;
  onPause: () => void;
  onRestart: () => void;
  onVolumeChange: (v: number) => void;
  onToggleMute: () => void;
}

export function AudioTransportControls({
  onPlay,
  onPause,
  onRestart,
  onVolumeChange,
  onToggleMute,
}: TransportProps) {
  const isPlaying = useStudioStore((s) => s.playback.isPlaying);
  const currentTime = useStudioStore((s) => s.playback.currentTime);
  const duration = useStudioStore((s) => s.playback.duration);
  const volume = useStudioStore((s) => s.playback.volume);
  const isMuted = useStudioStore((s) => s.playback.isMuted);
  const hasAudio = useStudioStore((s) => s.audioBuffer !== null);

  const disabled = !hasAudio;

  return (
    <div
      className="flex items-center gap-3 px-4 py-2.5 glass"
      style={{ opacity: disabled ? 0.4 : 1, transition: 'opacity 0.3s' }}
    >
      {/* Restart */}
      <button
        onClick={onRestart}
        disabled={disabled}
        className="p-1.5 rounded-[var(--radius-tag)] transition-all duration-150 cursor-pointer disabled:cursor-not-allowed"
        style={{ color: 'rgba(255,255,255,0.5)' }}
        onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.color = 'var(--foreground)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
        aria-label="Restart"
      >
        <RotateCcw size={16} strokeWidth={1.5} />
      </button>

      {/* Play/Pause */}
      <button
        onClick={isPlaying ? onPause : onPlay}
        disabled={disabled}
        className="p-2.5 rounded-[var(--radius-button)] transition-all duration-200 cursor-pointer disabled:cursor-not-allowed"
        style={{
          background: disabled ? 'rgba(255,255,255,0.04)' : 'var(--accent)',
          color: disabled ? 'rgba(255,255,255,0.3)' : '#0A0A0B',
        }}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <Pause size={18} strokeWidth={1.5} />
        ) : (
          <Play size={18} strokeWidth={1.5} style={{ marginLeft: 2 }} />
        )}
      </button>

      {/* Time */}
      <div className="flex items-center gap-1.5 font-mono text-xs tabular-nums min-w-[100px]">
        <span style={{ color: 'var(--accent)' }}>{formatTime(currentTime)}</span>
        <span style={{ color: 'rgba(255,255,255,0.2)' }}>/</span>
        <span style={{ color: 'rgba(255,255,255,0.4)' }}>{formatTime(duration)}</span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Volume */}
      <button
        onClick={onToggleMute}
        disabled={disabled}
        className="p-1.5 rounded-[var(--radius-tag)] transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed"
        style={{ color: isMuted ? '#FF2D7A' : 'rgba(255,255,255,0.5)' }}
        aria-label={isMuted ? 'Unmute' : 'Mute'}
      >
        {isMuted ? <VolumeX size={16} strokeWidth={1.5} /> : <Volume2 size={16} strokeWidth={1.5} />}
      </button>

      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={isMuted ? 0 : volume}
        onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-16 h-1 appearance-none rounded-full cursor-pointer disabled:cursor-not-allowed"
        style={{
          background: `linear-gradient(to right, var(--accent) ${(isMuted ? 0 : volume) * 100}%, rgba(255,255,255,0.1) ${(isMuted ? 0 : volume) * 100}%)`,
        }}
        aria-label="Volume"
      />
    </div>
  );
}
