import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { TimelineSection, PlaybackState, FrequencyData } from '@/types/audio';
import type { StyleMode, ControlValues, ExportState, BrowserCompat, DropzoneState, Toast, ToastVariant } from '@/types/studio';
import { DEFAULT_CONTROLS } from '@/types/studio';
import { generateId } from '@/lib/utils/format';

interface StudioState {
  // File slice — audioBuffer read only via getState(), never subscribed
  file: File | null;
  fileName: string | null;
  fileSize: number;
  audioBuffer: AudioBuffer | null;
  waveformPeaks: Float32Array | null;
  dropzoneState: DropzoneState;
  decodeError: string | null;

  // Playback slice
  playback: PlaybackState;

  // Mode slice (persisted)
  mode: StyleMode;

  // Controls slice (persisted)
  controls: ControlValues;

  // Sections
  sections: TimelineSection[];

  // Export
  exportState: ExportState;

  // Browser compat
  compat: BrowserCompat;

  // Toasts
  toasts: Toast[];

  // Hydration flag for SSR safety
  hasHydrated: boolean;

  // Actions
  setFile: (file: File) => void;
  setAudioBuffer: (buffer: AudioBuffer) => void;
  setWaveformPeaks: (peaks: Float32Array) => void;
  setDropzoneState: (state: DropzoneState) => void;
  setDecodeError: (error: string | null) => void;
  clearFile: () => void;

  setPlayback: (update: Partial<PlaybackState>) => void;
  togglePlay: () => void;
  setCurrentTime: (time: number) => void;

  setMode: (mode: StyleMode) => void;

  setControl: (key: keyof ControlValues, value: number) => void;
  resetControls: () => void;

  setSections: (sections: TimelineSection[]) => void;

  setExportState: (update: Partial<ExportState>) => void;
  resetExport: () => void;

  setCompat: (compat: Partial<BrowserCompat>) => void;

  addToast: (message: string, variant: ToastVariant, duration?: number) => void;
  removeToast: (id: string) => void;

  setHasHydrated: (value: boolean) => void;
}

export const useStudioStore = create<StudioState>()(
  persist(
    (set) => ({
      // File
      file: null,
      fileName: null,
      fileSize: 0,
      audioBuffer: null,
      waveformPeaks: null,
      dropzoneState: 'idle' as DropzoneState,
      decodeError: null,

      // Playback
      playback: {
        isPlaying: false,
        currentTime: 0,
        duration: 0,
        volume: 0.8,
        isMuted: false,
      },

      // Mode
      mode: 'alchemist' as StyleMode,

      // Controls
      controls: { ...DEFAULT_CONTROLS },

      // Sections
      sections: [],

      // Export
      exportState: {
        isRecording: false,
        recordingDuration: 0,
        recordingElapsed: 0,
        recordingProgress: 0,
      },

      // Compat
      compat: {
        hasMediaRecorder: false,
        supportedMimeType: null,
        hasAudioContext: false,
        isTouch: false,
        prefersReducedMotion: false,
      },

      // Toasts
      toasts: [],

      // Hydration
      hasHydrated: false,

      // File actions
      setFile: (file) => set({
        file,
        fileName: file.name,
        fileSize: file.size,
        dropzoneState: 'decoding',
        decodeError: null,
      }),
      setAudioBuffer: (buffer) => set({ audioBuffer: buffer }),
      setWaveformPeaks: (peaks) => set({ waveformPeaks: peaks }),
      setDropzoneState: (state) => set({ dropzoneState: state }),
      setDecodeError: (error) => set({ decodeError: error, dropzoneState: error ? 'error' : 'idle' }),
      clearFile: () => set({
        file: null,
        fileName: null,
        fileSize: 0,
        audioBuffer: null,
        waveformPeaks: null,
        dropzoneState: 'idle',
        decodeError: null,
        sections: [],
        playback: { isPlaying: false, currentTime: 0, duration: 0, volume: 0.8, isMuted: false },
      }),

      // Playback actions
      setPlayback: (update) => set((s) => ({ playback: { ...s.playback, ...update } })),
      togglePlay: () => set((s) => ({ playback: { ...s.playback, isPlaying: !s.playback.isPlaying } })),
      setCurrentTime: (time) => set((s) => ({ playback: { ...s.playback, currentTime: time } })),

      // Mode
      setMode: (mode) => set({ mode }),

      // Controls
      setControl: (key, value) => set((s) => ({
        controls: { ...s.controls, [key]: value },
      })),
      resetControls: () => set({ controls: { ...DEFAULT_CONTROLS } }),

      // Sections
      setSections: (sections) => set({ sections }),

      // Export
      setExportState: (update) => set((s) => ({
        exportState: { ...s.exportState, ...update },
      })),
      resetExport: () => set({
        exportState: {
          isRecording: false,
          recordingDuration: 0,
          recordingElapsed: 0,
          recordingProgress: 0,
        },
      }),

      // Compat
      setCompat: (compat) => set((s) => ({
        compat: { ...s.compat, ...compat },
      })),

      // Toasts
      addToast: (message, variant, duration = 4000) => {
        const id = generateId();
        set((s) => ({
          toasts: [...s.toasts.slice(-2), { id, message, variant, duration }],
        }));
        if (duration > 0) {
          setTimeout(() => {
            set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
          }, duration);
        }
      },
      removeToast: (id) => set((s) => ({
        toasts: s.toasts.filter((t) => t.id !== id),
      })),

      // Hydration
      setHasHydrated: (value) => set({ hasHydrated: value }),
    }),
    {
      name: 'ai-music-canvas-studio',
      skipHydration: true,
      partialize: (state) => ({
        mode: state.mode,
        controls: state.controls,
        playback: { ...state.playback, isPlaying: false, currentTime: 0 },
      }),
    }
  )
);
