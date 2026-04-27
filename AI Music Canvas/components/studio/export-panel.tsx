'use client';

import { useState, useCallback, useRef } from 'react';
import { Circle, Download, AlertTriangle } from 'lucide-react';
import { useStudioStore } from '@/store/studio-store';
import { startRecording, stopRecording, downloadBlob } from '@/lib/export/media-recorder';
import { getAudioContext } from '@/lib/audio/audio-context';

interface ExportPanelProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  gainRef: React.RefObject<GainNode | null>;
}

const DURATION_OPTIONS = [
  { label: '15s', value: 15000 },
  { label: '30s', value: 30000 },
  { label: 'Full', value: -1 },
];

export function ExportPanel({ canvasRef, gainRef }: ExportPanelProps) {
  const compat = useStudioStore((s) => s.compat);
  const isPlaying = useStudioStore((s) => s.playback.isPlaying);
  const duration = useStudioStore((s) => s.playback.duration);
  const exportState = useStudioStore((s) => s.exportState);
  const setExportState = useStudioStore((s) => s.setExportState);
  const resetExport = useStudioStore((s) => s.resetExport);
  const addToast = useStudioStore((s) => s.addToast);
  const hasAudio = useStudioStore((s) => s.audioBuffer !== null);

  const [selectedDuration, setSelectedDuration] = useState(15000);
  const blobRef = useRef<Blob | null>(null);

  const handleRecord = useCallback(() => {
    if (!canvasRef.current || !gainRef.current || !compat.supportedMimeType) return;

    const recordDuration = selectedDuration === -1 ? duration * 1000 : selectedDuration;

    setExportState({ isRecording: true, recordingDuration: recordDuration, recordingElapsed: 0 });

    startRecording({
      canvas: canvasRef.current,
      audioContext: getAudioContext(),
      gainNode: gainRef.current,
      mimeType: compat.supportedMimeType,
      durationMs: recordDuration,
      onProgress: (progress) => {
        setExportState({ recordingProgress: progress, recordingElapsed: progress * recordDuration });
      },
      onComplete: (blob) => {
        blobRef.current = blob;
        resetExport();
        addToast('Recording complete! Click download to save.', 'success');
      },
      onError: (error) => {
        resetExport();
        addToast(error, 'error');
      },
    });
  }, [canvasRef, gainRef, compat.supportedMimeType, selectedDuration, duration, setExportState, resetExport, addToast]);

  const handleStop = useCallback(() => {
    stopRecording();
  }, []);

  const handleDownload = useCallback(() => {
    if (blobRef.current) {
      downloadBlob(blobRef.current, `ai-music-canvas-${Date.now()}.webm`);
    }
  }, []);

  // Not supported
  if (!compat.hasMediaRecorder || !compat.supportedMimeType) {
    return (
      <div className="space-y-2">
        <p className="text-caption">Export</p>
        <div className="glass p-4 flex items-center gap-3">
          <AlertTriangle size={16} strokeWidth={1.5} style={{ color: '#E8B65A' }} />
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.5)' }}>
            Export requires Chrome or Firefox. Safari support is coming soon.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-caption">Export</p>

      {/* Duration selector */}
      <div className="flex gap-1.5">
        {DURATION_OPTIONS.map((opt) => (
          <button
            key={opt.label}
            onClick={() => setSelectedDuration(opt.value)}
            className="flex-1 py-1.5 text-xs font-medium rounded-[var(--radius-tag)] transition-all duration-150 cursor-pointer"
            style={{
              background: selectedDuration === opt.value ? 'rgba(var(--accent-rgb), 0.12)' : 'rgba(255,255,255,0.03)',
              color: selectedDuration === opt.value ? 'var(--accent)' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${selectedDuration === opt.value ? 'rgba(var(--accent-rgb), 0.2)' : 'rgba(255,255,255,0.04)'}`,
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Record/Stop */}
      {exportState.isRecording ? (
        <div className="space-y-2">
          <button
            onClick={handleStop}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-[var(--radius-button)] text-sm font-medium cursor-pointer transition-all duration-200"
            style={{
              background: 'rgba(255,45,122,0.12)',
              color: '#FF2D7A',
              border: '1px solid rgba(255,45,122,0.2)',
            }}
          >
            <Circle size={14} strokeWidth={1.5} className="animate-pulse" fill="currentColor" />
            Stop Recording
          </button>
          {/* Progress bar */}
          <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
            <div
              className="h-full rounded-full transition-[width] duration-100"
              style={{
                width: `${exportState.recordingProgress * 100}%`,
                background: '#FF2D7A',
              }}
            />
          </div>
        </div>
      ) : (
        <button
          onClick={handleRecord}
          disabled={!hasAudio || !isPlaying}
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-[var(--radius-button)] text-sm font-medium cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 transition-all duration-200"
          style={{
            background: 'rgba(var(--accent-rgb), 0.1)',
            color: 'var(--accent)',
            border: '1px solid rgba(var(--accent-rgb), 0.2)',
          }}
        >
          <Circle size={14} strokeWidth={1.5} />
          Record
        </button>
      )}

      {/* Download */}
      {blobRef.current && !exportState.isRecording && (
        <button
          onClick={handleDownload}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-[var(--radius-button)] text-xs font-medium cursor-pointer transition-all duration-200"
          style={{
            background: 'rgba(111,168,220,0.1)',
            color: '#6FA8DC',
            border: '1px solid rgba(111,168,220,0.2)',
          }}
        >
          <Download size={14} strokeWidth={1.5} />
          Download .webm
        </button>
      )}

      <p className="text-[10px]" style={{ color: 'rgba(255,255,255,0.25)' }}>
        Records at 30fps. Play audio first, then record.
      </p>
    </div>
  );
}
