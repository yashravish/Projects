/**
 * @swaram/voice - Carnatic music transcription library
 * 
 * Main entry point
 */

// Core pipeline
export { transcribe } from "./pipeline/transcribe";

// Export utilities
export { toMIDI } from "./export/midi";
export { toJSON, toJSONBytes, fromJSON, fromJSONBytes } from "./export/json";

// Theory utilities
export { SWARA_CENTS, centsToSwara, getSwaraWithOctave } from "./theory/swara-map";
export { RAGAS, getRaga, getAvailableRagas } from "./theory/ragas";
export { COMMON_TALAS, getTala } from "./theory/tala";

// Types
export type {
  Transcription,
  TranscribeOptions,
  SwaraEvent,
  NoteEvent,
  PitchFrame,
  Swara,
  TonicEstimate,
  AudioSourceInput,
  Hz,
  Cents,
  TimeSec,
} from "./types";

export type { RagaDefinition } from "./theory/ragas";
export type { TalaDefinition } from "./theory/tala";
