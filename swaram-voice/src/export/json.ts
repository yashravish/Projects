/**
 * JSON export functionality
 */

import type { Transcription } from "../types";

/**
 * Export transcription to JSON string
 */
export function toJSON(tx: Transcription, pretty: boolean = true): string {
  return JSON.stringify(tx, null, pretty ? 2 : 0);
}

/**
 * Export transcription to JSON bytes
 */
export function toJSONBytes(tx: Transcription, pretty: boolean = true): Uint8Array {
  const json = toJSON(tx, pretty);
  const encoder = new TextEncoder();
  return encoder.encode(json);
}

/**
 * Parse transcription from JSON string
 */
export function fromJSON(json: string): Transcription {
  return JSON.parse(json) as Transcription;
}

/**
 * Parse transcription from JSON bytes
 */
export function fromJSONBytes(bytes: Uint8Array): Transcription {
  const decoder = new TextDecoder();
  const json = decoder.decode(bytes);
  return fromJSON(json);
}
