/**
 * Main transcription pipeline
 */

import type { Transcription, TranscribeOptions, PitchFrame } from "../types";
import { createAudioSource } from "../audio/AudioSource";
import { resample } from "../audio/resample";
import { hann, applyWindow } from "../audio/windowing";
import { YINDetector } from "../dsp/pitch/yin";
import { ACFDetector } from "../dsp/pitch/acf";
import type { PitchDetector } from "../dsp/pitch/types";
import { medianFilter, fillGaps } from "../dsp/smoothing";
import { detectTonic, validateTonic } from "../theory/tonic";
import { detectTempoFromAudio } from "../theory/tala";
import { segmentNotes, notesToSwaras, mergeSwaras, filterShortSwaras } from "./postprocess";

/**
 * Main transcription function
 */
export async function transcribe(
  audio: Float32Array | AudioBuffer | ArrayBuffer | Buffer,
  opts: TranscribeOptions = {}
): Promise<Transcription> {
  // Step 1: Decode and resample audio
  // If caller provided Float32Array plus an explicit sampleRate option, respect it
  const explicitRate = opts.sampleRate;
  const audioSource =
    audio instanceof Float32Array && typeof explicitRate === "number"
      ? {
          sampleRate: explicitRate,
          samples: audio,
          duration: audio.length / explicitRate,
        }
      : await createAudioSource(audio);
  const targetSampleRate = opts.sampleRate ?? 22050;
  
  let samples = audioSource.samples;
  if (audioSource.sampleRate !== targetSampleRate) {
    samples = resample(samples, audioSource.sampleRate, targetSampleRate);
  }

  // Step 2: Extract pitch frames
  const frames = extractPitchFrames(samples, targetSampleRate, opts);

  // Step 3: Smooth pitch track
  const smoothed = medianFilter(frames, 5);
  const filled = fillGaps(smoothed, 100, targetSampleRate);

  // Step 4: Detect or use provided tonic
  let tonic;
  if (opts.tonicHz && opts.tonicHz !== "auto") {
    tonic = {
      hz: opts.tonicHz,
      confidence: 1.0,
    };
  } else {
    tonic = detectTonic(filled);
    
    // Validate tonic
    const validationScore = validateTonic(filled, tonic.hz);
    tonic.confidence *= validationScore;

    // Final clamp: prefer the lowest stable voiced f0 as tonic when auto-picked value is an octave high
    const voicedF0s = filled.filter((f) => f.f0 !== null && f.voicing > 0.5).map((f) => f.f0!) as number[];
    if (voicedF0s.length > 0) {
      const minF0 = Math.min(...voicedF0s);
      if (tonic.hz > minF0 * 1.1) {
        tonic.hz = minF0;
      }
    }
  }

  // Step 5: Segment into notes
  const minNoteEnergy = opts.minNoteEnergy ?? 0.01;
  const notes = segmentNotes(filled, 0.08, 15, minNoteEnergy);

  // Step 6: Map notes to swaras
  const snapToleranceCents = opts.snapToleranceCents ?? 50;
  let swaras = notesToSwaras(notes, tonic, opts.ragaHint ?? null, snapToleranceCents);

  // Step 7: Post-process swaras
  swaras = mergeSwaras(swaras, 0.1);
  swaras = filterShortSwaras(swaras, 0.05);

  // Step 8: Detect tempo (optional)
  let tempo: number | undefined;
  if (opts.tempoBPM === "auto") {
    const detectedTempo = detectTempoFromAudio(samples, targetSampleRate);
    tempo = detectedTempo ?? undefined;
  } else if (opts.tempoBPM) {
    tempo = opts.tempoBPM;
  }

  return {
    tonic,
    swaras,
    notes,
    tempo,
  };
}

/**
 * Extract pitch frames from audio using sliding window
 */
function extractPitchFrames(
  samples: Float32Array,
  sampleRate: number,
  opts: TranscribeOptions
): PitchFrame[] {
  const frameSize = opts.frameSize ?? 2048;
  const hopSize = opts.hopSize ?? 512;
  const detectorType = opts.detector ?? "yin";
  const minVoicing = opts.minVoicing ?? 0.3;

  // Create pitch detector
  const detector: PitchDetector =
    detectorType === "acf"
      ? new ACFDetector({ minFreq: 80, maxFreq: 800 })
      : new YINDetector({ minFreq: 80, maxFreq: 800 });

  // Generate window function
  const window = hann(frameSize);

  const frames: PitchFrame[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const end = start + frameSize;

    // Extract frame
    const frame = samples.slice(start, end);

    // Apply window
    const windowed = applyWindow(frame, window);

    // Detect pitch
    const result = detector.detect(windowed, sampleRate);

    // Filter by voicing threshold
    const f0 = result.voicing >= minVoicing ? result.f0 : null;

    frames.push({
      t: (start + frameSize / 2) / sampleRate,
      f0,
      voicing: result.voicing,
    });
  }

  return frames;
}
