import { CONFIG } from '@signaldesk/shared';

export function chunkText(text: string): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + CONFIG.CHUNK_SIZE_CHARS, text.length);
    chunks.push(text.slice(start, end));

    start = end - CONFIG.CHUNK_OVERLAP_CHARS;
    if (start + CONFIG.CHUNK_OVERLAP_CHARS >= text.length) break;
  }

  return chunks;
}

export function estimateTokenCount(text: string): number {
  return Math.ceil(text.length / CONFIG.CHARS_PER_TOKEN_ESTIMATE);
}
