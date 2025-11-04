/**
 * Core type definitions for @swaram/voice
 */

export type TimeSec = number;
export type Hz = number;
export type Cents = number;

/**
 * A single pitch detection frame
 */
export interface PitchFrame {
  /** Center time of frame in seconds */
  t: TimeSec;
  /** Fundamental frequency in Hz, or null if unvoiced */
  f0: Hz | null;
  /** Voicing confidence 0..1 */
  voicing: number;
}

/**
 * A segmented note event from continuous voiced regions
 */
export interface NoteEvent {
  /** Start time in seconds */
  start: TimeSec;
  /** End time in seconds */
  end: TimeSec;
  /** Average f0 in this segment */
  f0Mean: Hz;
  /** Coarse gamaka indicator (true if high slope/variance) */
  glide: boolean;
}

/**
 * Carnatic swara names (L0 set)
 */
export type Swara =
  | "S"
  | "R1"
  | "R2"
  | "G2"
  | "G3"
  | "M1"
  | "M2"
  | "P"
  | "D1"
  | "D2"
  | "N2"
  | "N3";

/**
 * A swara event with timing and pitch information
 */
export interface SwaraEvent {
  /** Start time in seconds */
  start: TimeSec;
  /** End time in seconds */
  end: TimeSec;
  /** Identified swara */
  swara: Swara;
  /** Cents offset from Sa (tonic) */
  centsFromSa: Cents;
  /** Confidence score 0..1 */
  confidence: number;
  /** Optional glide indicator */
  glide?: boolean;
}

/**
 * Tonic (Sa) detection result
 */
export interface TonicEstimate {
  /** Tonic frequency in Hz */
  hz: Hz;
  /** Confidence score 0..1 */
  confidence: number;
}

/**
 * Configuration options for transcription
 */
export interface TranscribeOptions {
  /** Target sample rate (default: 22050). Note: Float32Array input defaults to 44100 if not resampled. */
  sampleRate?: number;
  /** FFT/analysis frame size (default: 2048) */
  frameSize?: number;
  /** Hop size between frames (default: 512) */
  hopSize?: number;
  /** Pitch detector algorithm (default: "yin") */
  detector?: "yin" | "acf";
  /** Optional raga hint for swara mapping */
  ragaHint?: string | null;
  /** Tempo in BPM, "auto", or null to skip */
  tempoBPM?: number | "auto" | null;
  /** Tonic frequency in Hz, or "auto" to detect */
  tonicHz?: Hz | "auto";
  /** Minimum voicing confidence to accept frames (default: 0.3) */
  minVoicing?: number;
  /** Snap tolerance in cents for swara mapping (default: 50) */
  snapToleranceCents?: number;
  /** Minimum RMS energy for note detection (default: 0.01) */
  minNoteEnergy?: number;
}

/**
 * Complete transcription result
 */
export interface Transcription {
  /** Detected or provided tonic */
  tonic: TonicEstimate;
  /** Array of swara events */
  swaras: SwaraEvent[];
  /** Intermediate note events */
  notes: NoteEvent[];
  /** Detected tempo in BPM (if applicable) */
  tempo?: number;
}

/**
 * Internal audio source interface
 */
export interface AudioSource {
  /** Sample rate in Hz */
  sampleRate: number;
  /** Audio samples (mono) */
  samples: Float32Array;
  /** Duration in seconds */
  duration: number;
}

/**
 * Alternative input format with explicit sample rate
 */
export interface AudioSourceInput {
  samples: Float32Array;
  sampleRate: number;
}
