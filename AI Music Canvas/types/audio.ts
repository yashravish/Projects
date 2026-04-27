export interface AudioFileData {
  file: File;
  name: string;
  size: number;
  duration: number;
  sampleRate: number;
  numberOfChannels: number;
}

export interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
}

export type SectionType = 'intro' | 'verse' | 'chorus' | 'bridge' | 'outro';

export interface TimelineSection {
  id: string;
  type: SectionType;
  startTime: number;
  endTime: number;
  label: string;
  color: string;
}

export interface WaveformData {
  peaks: Float32Array;
  length: number;
  duration: number;
}

export interface FrequencyData {
  frequency: Uint8Array;
  timeDomain: Uint8Array;
  bass: number;
  mid: number;
  treble: number;
  energy: number;
}
