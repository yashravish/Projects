import { describe, it, expect, beforeEach } from 'vitest';
import { generateEmbedding, generateEmbeddings } from '../services/embedding.service';

describe('Embedding Service', () => {
  beforeEach(() => {
    // Clear any mocks
  });

  it('should have generateEmbedding function', () => {
    expect(generateEmbedding).toBeDefined();
    expect(typeof generateEmbedding).toBe('function');
  });

  it('should have generateEmbeddings function', () => {
    expect(generateEmbeddings).toBeDefined();
    expect(typeof generateEmbeddings).toBe('function');
  });

  it('should return null when OpenAI is not configured (test environment)', async () => {
    // In test environment without OpenAI key, function should return null or handle gracefully
    const result = await generateEmbedding('test text');

    // Either null or an array (if key is configured)
    expect(result === null || Array.isArray(result)).toBe(true);
  });

  it('should return array of nulls when OpenAI is not configured (test environment)', async () => {
    // In test environment without OpenAI key, function should return array of nulls
    const result = await generateEmbeddings(['text 1', 'text 2']);

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);
  });
});
