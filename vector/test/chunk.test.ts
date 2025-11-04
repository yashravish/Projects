import { describe, it, expect } from 'vitest';
import { chunkText } from '../src/ingest/chunk.js';

describe('chunkText', () => {
  it('splits markdown by sections and respects overlap', () => {
    const md = '# A\n\npara1\n\n## B\n\npara2\n';
    const chunks = chunkText('docs/readme.md', md, 'md', { targetChunkSize: 10, overlap: 5 });
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].path).toBe('docs/readme.md');
    expect(chunks[0].startLine).toBe(1);
  });
});


