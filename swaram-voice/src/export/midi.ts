/**
 * MIDI export functionality
 */

import type { Transcription, Swara } from "../types";
import { SWARA_CENTS } from "../theory/swara-map";

/**
 * Convert transcription to Standard MIDI File (Format 0, single track)
 */
export function toMIDI(tx: Transcription, baseMIDINote: number = 60): Uint8Array {
  // Use C4 (MIDI note 60) as Sa by default
  const ticksPerQuarter = 480;
  const microsecondsPerQuarter = 500000; // 120 BPM default

  // Build MIDI events
  const events: MIDIEvent[] = [];

  // Add tempo event
  events.push({
    deltaTime: 0,
    type: "meta",
    metaType: 0x51, // Set Tempo
    data: new Uint8Array([
      (microsecondsPerQuarter >> 16) & 0xff,
      (microsecondsPerQuarter >> 8) & 0xff,
      microsecondsPerQuarter & 0xff,
    ]),
  });

  // Convert swaras to MIDI notes
  for (const swara of tx.swaras) {
    const midiNote = swaraToMIDINote(swara.swara, baseMIDINote);
    const startTick = Math.round((swara.start * 1000000 * ticksPerQuarter) / microsecondsPerQuarter);
    const endTick = Math.round((swara.end * 1000000 * ticksPerQuarter) / microsecondsPerQuarter);

    // Note On
    events.push({
      deltaTime: startTick,
      type: "channel",
      status: 0x90, // Note On
      channel: 0,
      data: new Uint8Array([midiNote, 80]), // velocity 80
    });

    // Note Off
    events.push({
      deltaTime: endTick,
      type: "channel",
      status: 0x80, // Note Off
      channel: 0,
      data: new Uint8Array([midiNote, 0]),
    });
  }

  // Sort events by time
  events.sort((a, b) => a.deltaTime - b.deltaTime);

  // Convert absolute times to delta times
  let lastTime = 0;
  for (const event of events) {
    const absoluteTime = event.deltaTime;
    event.deltaTime = absoluteTime - lastTime;
    lastTime = absoluteTime;
  }

  // End of track
  events.push({
    deltaTime: 0,
    type: "meta",
    metaType: 0x2f,
    data: new Uint8Array([]),
  });

  // Build MIDI file
  return buildMIDIFile(events, ticksPerQuarter);
}

interface MIDIEvent {
  deltaTime: number;
  type: "channel" | "meta";
  status?: number;
  channel?: number;
  metaType?: number;
  data: Uint8Array;
}

/**
 * Convert swara to MIDI note number
 */
function swaraToMIDINote(swara: Swara, baseMIDINote: number): number {
  const cents = SWARA_CENTS[swara];
  const semitones = Math.round(cents / 100);
  return baseMIDINote + semitones;
}

/**
 * Build complete MIDI file bytes
 */
function buildMIDIFile(events: MIDIEvent[], ticksPerQuarter: number): Uint8Array {
  // Build track chunk
  const trackData: number[] = [];

  for (const event of events) {
    // Write variable-length delta time
    writeVarLen(trackData, event.deltaTime);

    if (event.type === "meta") {
      // Meta event
      trackData.push(0xff, event.metaType!);
      writeVarLen(trackData, event.data.length);
      trackData.push(...event.data);
    } else {
      // Channel event
      trackData.push((event.status! | event.channel!));
      trackData.push(...event.data);
    }
  }

  // Build header chunk
  const header = new Uint8Array([
    // "MThd"
    0x4d,
    0x54,
    0x68,
    0x64,
    // Chunk length (6 bytes)
    0x00,
    0x00,
    0x00,
    0x06,
    // Format 0
    0x00,
    0x00,
    // Number of tracks (1)
    0x00,
    0x01,
    // Ticks per quarter note
    (ticksPerQuarter >> 8) & 0xff,
    ticksPerQuarter & 0xff,
  ]);

  // Build track chunk header
  const trackHeader = new Uint8Array([
    // "MTrk"
    0x4d,
    0x54,
    0x72,
    0x6b,
    // Chunk length (4 bytes)
    (trackData.length >> 24) & 0xff,
    (trackData.length >> 16) & 0xff,
    (trackData.length >> 8) & 0xff,
    trackData.length & 0xff,
  ]);

  // Combine all parts
  const result = new Uint8Array(header.length + trackHeader.length + trackData.length);
  result.set(header, 0);
  result.set(trackHeader, header.length);
  result.set(trackData, header.length + trackHeader.length);

  return result;
}

/**
 * Write variable-length quantity
 */
function writeVarLen(buffer: number[], value: number): void {
  if (value < 0) {
    value = 0;
  }

  const bytes: number[] = [];
  bytes.push(value & 0x7f);

  while (value > 0x7f) {
    value >>= 7;
    bytes.push((value & 0x7f) | 0x80);
  }

  // Reverse and write
  for (let i = bytes.length - 1; i >= 0; i--) {
    buffer.push(bytes[i]);
  }
}
