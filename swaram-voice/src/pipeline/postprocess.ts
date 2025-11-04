/**
 * Post-processing utilities for note segmentation and gamaka detection
 */

import type { Hz, NoteEvent, PitchFrame, SwaraEvent, Swara, TonicEstimate } from "../types";
import { centsToSwara, ALL_SWARAS } from "../theory/swara-map";
import { getRaga } from "../theory/ragas";

/**
 * Segment continuous voiced regions into note events
 */
export function segmentNotes(
  frames: PitchFrame[],
  minNoteLengthSec: number = 0.08,
  stabilityThreshold: number = 15, // cents
  minNoteEnergy: number = 0.01
): NoteEvent[] {
  const notes: NoteEvent[] = [];
  let currentNote: {
    start: number;
    frames: PitchFrame[];
  } | null = null;

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];

    if (frame.f0 !== null && frame.voicing > 0.5) {
      if (!currentNote) {
        // Start new note
        currentNote = {
          start: frame.t,
          frames: [frame],
        };
      } else {
        // Check for extreme outliers (likely pitch detection errors)
        if (currentNote.frames.length > 0) {
          const prevF0 = currentNote.frames[currentNote.frames.length - 1].f0!;
          const cents = Math.abs(1200 * Math.log2(frame.f0 / prevF0));
          
          // If jump > 1200 cents (more than an octave), treat as new note
          if (cents > 1200) {
            const note = finalizeNote(currentNote, minNoteLengthSec, stabilityThreshold, minNoteEnergy);
            if (note) {
              notes.push(note);
            }
            currentNote = {
              start: frame.t,
              frames: [frame],
            };
            continue;
          }
        }
        
        // Continue current note
        currentNote.frames.push(frame);
      }
    } else {
      // End current note if it exists
      if (currentNote && currentNote.frames.length > 0) {
        const note = finalizeNote(currentNote, minNoteLengthSec, stabilityThreshold, minNoteEnergy);
        if (note) {
          notes.push(note);
        }
        currentNote = null;
      }
    }
  }

  // Finalize last note
  if (currentNote && currentNote.frames.length > 0) {
    const note = finalizeNote(currentNote, minNoteLengthSec, stabilityThreshold, minNoteEnergy);
    if (note) {
      notes.push(note);
    }
  }

  return notes;
}

/**
 * Finalize a note from accumulated frames
 */
function finalizeNote(
  currentNote: { start: number; frames: PitchFrame[] },
  minNoteLengthSec: number,
  stabilityThreshold: number,
  minNoteEnergy: number
): NoteEvent | null {
  const frames = currentNote.frames;
  const end = frames[frames.length - 1].t;
  const duration = end - currentNote.start;

  // Skip notes that are too short
  if (duration < minNoteLengthSec) {
    return null;
  }

  // Calculate mean f0
  const f0Values = frames.map((f) => f.f0!);
  const f0Mean = f0Values.reduce((sum, val) => sum + val, 0) / f0Values.length;

  // Check minimum energy (using voicing as proxy for energy)
  const avgVoicing = frames.reduce((sum, f) => sum + f.voicing, 0) / frames.length;
  if (avgVoicing < minNoteEnergy) {
    return null;
  }

  // Detect glide (high variance or slope)
  const glide = isGlide(f0Values, stabilityThreshold);

  return {
    start: currentNote.start,
    end,
    f0Mean,
    glide,
  };
}

/**
 * Detect if a note has significant glide/gamaka
 */
function isGlide(f0Values: Hz[], thresholdCents: number): boolean {
  if (f0Values.length < 3) {
    return false;
  }

  // Calculate variance in cents
  const meanF0 = f0Values.reduce((sum, val) => sum + val, 0) / f0Values.length;
  const centsValues = f0Values.map((f0) => 1200 * Math.log2(f0 / meanF0));
  
  const variance =
    centsValues.reduce((sum, cents) => sum + Math.pow(cents, 2), 0) / centsValues.length;
  const stdDev = Math.sqrt(variance);

  // Check if standard deviation exceeds threshold
  if (stdDev > thresholdCents) {
    return true;
  }

  // Also check for monotonic slope (ascending or descending glide)
  const slope = (f0Values[f0Values.length - 1] - f0Values[0]) / f0Values.length;
  const slopeCents = 1200 * Math.log2(Math.abs(slope) + 1);

  return slopeCents > thresholdCents / 2;
}

/**
 * Map notes to swaras
 */
export function notesToSwaras(
  notes: NoteEvent[],
  tonic: TonicEstimate,
  ragaHint: string | null = null,
  snapToleranceCents: number = 50
): SwaraEvent[] {
  const swaras: SwaraEvent[] = [];

  // Get allowed swaras from raga hint
  let allowedSwaras: Swara[] = ALL_SWARAS;
  let useRagaFallback = false;
  
  if (ragaHint) {
    const raga = getRaga(ragaHint);
    if (raga) {
      allowedSwaras = raga.allowedSwaras;
      useRagaFallback = true;
    }
  }

  for (const note of notes) {
    // Convert to cents from Sa
    const centsFromSa = 1200 * Math.log2(note.f0Mean / tonic.hz);

    // Try to map to swara with raga constraints
    let result = centsToSwara(centsFromSa, allowedSwaras, snapToleranceCents);

    // If raga-constrained mapping failed and confidence is poor, fallback to global
    if (!result && useRagaFallback) {
      result = centsToSwara(centsFromSa, ALL_SWARAS, snapToleranceCents);
      // Mark with lower confidence since it's out-of-raga
      if (result) {
        result.confidence *= 0.5;
      }
    }

    if (result) {
      swaras.push({
        start: note.start,
        end: note.end,
        swara: result.swara,
        centsFromSa: result.cents,
        confidence: result.confidence * tonic.confidence,
        glide: note.glide,
      });
    }
  }

  return swaras;
}

/**
 * Merge consecutive identical swaras
 */
export function mergeSwaras(swaras: SwaraEvent[], maxGapSec: number = 0.1): SwaraEvent[] {
  if (swaras.length === 0) {
    return [];
  }

  const merged: SwaraEvent[] = [];
  let current = { ...swaras[0] };

  for (let i = 1; i < swaras.length; i++) {
    const next = swaras[i];

    // Merge if same swara and close in time
    if (
      next.swara === current.swara &&
      next.start - current.end <= maxGapSec &&
      (current.glide === next.glide || (!current.glide && !next.glide))
    ) {
      // Extend current swara
      current.end = next.end;
      current.confidence = Math.max(current.confidence, next.confidence);
      current.glide = current.glide || next.glide;
    } else {
      // Push current and start new
      merged.push(current);
      current = { ...next };
    }
  }

  // Push last swara
  merged.push(current);

  return merged;
}

/**
 * Filter out very short swaras (likely artifacts)
 */
export function filterShortSwaras(swaras: SwaraEvent[], minDurationSec: number = 0.05): SwaraEvent[] {
  return swaras.filter((swara) => swara.end - swara.start >= minDurationSec);
}
