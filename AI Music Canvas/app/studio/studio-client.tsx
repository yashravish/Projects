'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useStudioStore } from '@/store/studio-store';
import { useMediaQuery } from '@/hooks/use-media-query';
import { useAudioAnalyzer } from '@/hooks/use-audio-analyzer';
import { useKeyboardShortcuts } from '@/hooks/use-keyboard-shortcuts';
import { detectMediaRecorderCompat } from '@/lib/export/compat';
import { isAudioContextSupported } from '@/lib/audio/compat';
import { applyModeTokens } from '@/lib/styles/theme-tokens';
import { AudioDropzone } from '@/components/studio/audio-dropzone';
import { AudioTransportControls } from '@/components/studio/audio-transport-controls';
import { VisualCanvas } from '@/components/studio/visual-canvas';
import { WaveformTimeline } from '@/components/studio/waveform-timeline';
import { StyleModeSelector } from '@/components/studio/style-mode-selector';
import { ControlPanel } from '@/components/studio/control-panel';
import { ExportPanel } from '@/components/studio/export-panel';
import { KeyboardShortcutsModal } from '@/components/studio/keyboard-shortcuts-modal';
import { Keyboard } from 'lucide-react';

export default function StudioClient() {
  const { prefersReducedMotion, isTouch, isMobile } = useMediaQuery();
  const mode = useStudioStore((s) => s.mode);
  const audioBuffer = useStudioStore((s) => s.audioBuffer);
  const setCompat = useStudioStore((s) => s.setCompat);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const {
    getFrequencyData,
    gainRef,
    play,
    pause,
    seek,
    setVolume,
    toggleMute,
    pauseTimeRef,
  } = useAudioAnalyzer();

  const handlePlay = useCallback(() => {
    const buffer = useStudioStore.getState().audioBuffer;
    if (!buffer) return;
    const { isPlaying, currentTime } = useStudioStore.getState().playback;
    if (isPlaying) {
      pause();
    } else {
      play(buffer, currentTime || pauseTimeRef.current);
    }
  }, [play, pause, pauseTimeRef]);

  const handleSeekForward = useCallback(() => {
    const { currentTime, duration } = useStudioStore.getState().playback;
    seek(Math.min(currentTime + 5, duration));
  }, [seek]);

  const handleSeekBackward = useCallback(() => {
    const { currentTime } = useStudioStore.getState().playback;
    seek(Math.max(currentTime - 5, 0));
  }, [seek]);

  const handleRestart = useCallback(() => {
    const buffer = useStudioStore.getState().audioBuffer;
    if (buffer) {
      seek(0);
      play(buffer, 0);
    }
  }, [seek, play]);

  const { showShortcuts, setShowShortcuts } = useKeyboardShortcuts({
    onPlay: handlePlay,
    onSeekForward: handleSeekForward,
    onSeekBackward: handleSeekBackward,
  });

  // Detect browser capabilities on mount
  useEffect(() => {
    const recorderCompat = detectMediaRecorderCompat();
    setCompat({
      hasMediaRecorder: recorderCompat.supported,
      supportedMimeType: recorderCompat.mimeType,
      hasAudioContext: isAudioContextSupported(),
      isTouch,
      prefersReducedMotion,
    });
  }, [setCompat, isTouch, prefersReducedMotion]);

  // Apply mode theme tokens
  useEffect(() => {
    applyModeTokens(mode);
  }, [mode]);

  // Mobile fallback
  if (isMobile) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8 pt-20">
        <div className="glass p-8 text-center max-w-sm space-y-4">
          <p className="font-display text-2xl" style={{ color: 'var(--accent)' }}>
            Best on desktop
          </p>
          <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
            AI Music Canvas is an interactive audio visualization studio designed for larger screens.
            Visit on a desktop or tablet for the full experience.
          </p>
        </div>
      </div>
    );
  }

  const hasAudio = audioBuffer !== null;

  return (
    <div className="h-screen flex flex-col pt-14 overflow-hidden" style={{ background: 'var(--mode-bg)' }}>
      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left rail: Timeline + Transport */}
        <div
          className="w-64 flex-shrink-0 flex flex-col p-3 gap-3 overflow-y-auto"
          style={{ borderRight: '1px solid rgba(255,255,255,0.04)' }}
        >
          <AudioDropzone />

          {hasAudio && (
            <>
              <WaveformTimeline onSeek={seek} />
              <AudioTransportControls
                onPlay={() => {
                  const buffer = useStudioStore.getState().audioBuffer;
                  if (buffer) play(buffer, pauseTimeRef.current);
                }}
                onPause={pause}
                onRestart={handleRestart}
                onVolumeChange={setVolume}
                onToggleMute={toggleMute}
              />
            </>
          )}
        </div>

        {/* Center: Canvas */}
        <div className="flex-1 relative">
          <VisualCanvas
            getFrequencyData={getFrequencyData}
            reducedMotion={prefersReducedMotion}
            isTouch={isTouch}
          />
        </div>

        {/* Right rail: Controls */}
        <div
          className="w-60 flex-shrink-0 flex flex-col p-3 gap-4 overflow-y-auto"
          style={{ borderLeft: '1px solid rgba(255,255,255,0.04)' }}
        >
          <StyleModeSelector />
          <ControlPanel />
          <ExportPanel canvasRef={canvasRef} gainRef={gainRef} />

          {/* Shortcuts hint */}
          <button
            onClick={() => setShowShortcuts(true)}
            className="flex items-center gap-2 text-[10px] uppercase tracking-wider px-2 py-1.5 rounded-[var(--radius-tag)] transition-colors duration-200 cursor-pointer mt-auto"
            style={{ color: 'rgba(255,255,255,0.2)' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.2)'; }}
            aria-label="Show keyboard shortcuts"
          >
            <Keyboard size={14} strokeWidth={1.5} />
            Shortcuts (?)
          </button>
        </div>
      </div>

      <KeyboardShortcutsModal isOpen={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </div>
  );
}
