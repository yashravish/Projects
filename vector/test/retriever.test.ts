import { describe, it, expect } from 'vitest';
import { mmrDiversify } from '../src/retrieval/retriever.js';

function f32(arr: number[]): Float32Array { return new Float32Array(arr); }

describe('mmrDiversify', () => {
  it('returns up to k diversified items', () => {
    const candidates = [
      { vec: f32([1,0]), item: { record: { id: '1', path: 'a', startLine: 1, endLine: 1, content: '', lang: 'ts', ingestedAt: Date.now() }, score: 0.9 } },
      { vec: f32([0,1]), item: { record: { id: '2', path: 'b', startLine: 1, endLine: 1, content: '', lang: 'ts', ingestedAt: Date.now() }, score: 0.85 } },
      { vec: f32([1,0]), item: { record: { id: '3', path: 'c', startLine: 1, endLine: 1, content: '', lang: 'ts', ingestedAt: Date.now() }, score: 0.8 } },
    ];
    const out = mmrDiversify(candidates, 2, 0.5);
    expect(out.length).toBe(2);
  });
});


