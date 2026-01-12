import { describe, it, expect } from 'vitest';
import { chunkText, estimateTokenCount } from '../services/chunking.service';

describe('Chunking Service', () => {
  it('should chunk text correctly with overlap', () => {
    const text = 'A'.repeat(3000);
    const chunks = chunkText(text);

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].length).toBe(1200);
  });

  it('should handle short text', () => {
    const text = 'Short text';
    const chunks = chunkText(text);

    expect(chunks.length).toBe(1);
    expect(chunks[0]).toBe(text);
  });

  it('should estimate token count', () => {
    const text = 'This is a test';
    const tokenCount = estimateTokenCount(text);

    expect(tokenCount).toBeGreaterThan(0);
    expect(tokenCount).toBe(Math.ceil(text.length / 4));
  });
});
