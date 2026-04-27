export type StyleMode = 'alchemist' | 'ambient' | 'trap' | 'orchestral';

export interface ControlValues {
  intensity: number;
  particleCount: number;
  blur: number;
  glow: number;
  waveformThickness: number;
  backgroundMotion: number;
}

export const DEFAULT_CONTROLS: ControlValues = {
  intensity: 50,
  particleCount: 500,
  blur: 4,
  glow: 50,
  waveformThickness: 3,
  backgroundMotion: 50,
};

export interface ExportState {
  isRecording: boolean;
  recordingDuration: number;
  recordingElapsed: number;
  recordingProgress: number;
}

export interface BrowserCompat {
  hasMediaRecorder: boolean;
  supportedMimeType: string | null;
  hasAudioContext: boolean;
  isTouch: boolean;
  prefersReducedMotion: boolean;
}

export type DropzoneState = 'idle' | 'hover-armed' | 'drag-over' | 'decoding' | 'analyzing' | 'success' | 'error';

export type ToastVariant = 'success' | 'error' | 'info';

export interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
  duration?: number;
}
