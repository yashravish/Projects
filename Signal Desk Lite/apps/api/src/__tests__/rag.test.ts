import { describe, it, expect, vi, beforeEach } from 'vitest';
import { performRAGQuery } from '../services/rag.service';

// Mock dependencies
vi.mock('../db', () => ({
  prisma: {
    $queryRaw: vi.fn(),
    document: {
      findMany: vi.fn(),
    },
    documentChunk: {
      findMany: vi.fn(),
    },
  },
}));

vi.mock('../services/embedding.service', () => ({
  generateEmbedding: vi.fn(),
}));

vi.mock('openai', () => ({
  default: vi.fn(),
}));

vi.mock('../config', () => ({
  config: {
    openaiApiKey: 'test-key',
  },
}));

describe('RAG Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it('should return stub response when OpenAI is not configured', async () => {
    vi.doMock('../config', () => ({
      config: {
        openaiApiKey: null,
      },
    }));

    const { generateEmbedding } = await import('../services/embedding.service');
    vi.mocked(generateEmbedding).mockResolvedValue(null);

    const { prisma } = await import('../db');
    vi.mocked(prisma.documentChunk.findMany).mockResolvedValue([
      {
        id: 'chunk-1',
        content: 'Test content',
        documentId: 'doc-1',
        chunkIndex: 0,
        userId: 'user-1',
        tokenCount: 10,
        embedding: null,
        createdAt: new Date(),
      },
    ]);

    vi.mocked(prisma.document.findMany).mockResolvedValue([
      {
        id: 'doc-1',
        originalName: 'test.pdf',
      } as any,
    ]);

    const { performRAGQuery: ragQuery } = await import('../services/rag.service');

    const result = await ragQuery('test question', 'collection-1', 'user-1');

    expect(result.answer).toContain('stub response');
    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.usage.totalTokens).toBe(0);
  });

  it('should perform RAG query with embeddings', async () => {
    vi.doMock('../config', () => ({
      config: {
        openaiApiKey: 'test-key',
      },
    }));

    const mockEmbedding = new Array(1536).fill(0.1);

    const { generateEmbedding } = await import('../services/embedding.service');
    vi.mocked(generateEmbedding).mockResolvedValue(mockEmbedding);

    const { prisma } = await import('../db');
    vi.mocked(prisma.$queryRaw).mockResolvedValue([
      {
        id: 'chunk-1',
        content: 'Relevant content from document',
        documentId: 'doc-1',
        chunkIndex: 0,
        score: 0.95,
      },
    ]);

    vi.mocked(prisma.document.findMany).mockResolvedValue([
      {
        id: 'doc-1',
        originalName: 'test.pdf',
      } as any,
    ]);

    const OpenAI = (await import('openai')).default;
    const mockCompletion = vi.fn().mockResolvedValue({
      choices: [
        {
          message: {
            content: JSON.stringify({
              answer: 'This is the answer based on the context [1]',
              keyFacts: [
                { fact: 'Fact 1', source: 'chunk-1' },
                { fact: 'Fact 2', source: 'chunk-1' },
              ],
              summary: 'This is a summary of the answer.',
            }),
          },
        },
      ],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
      },
    });

    // @ts-ignore
    OpenAI.mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCompletion,
        },
      },
      embeddings: {
        create: vi.fn().mockResolvedValue({
          data: [{ embedding: mockEmbedding }],
        }),
      },
    }));

    const { performRAGQuery: ragQuery } = await import('../services/rag.service');

    const result = await ragQuery('What is the main topic?', 'collection-1', 'user-1');

    expect(result.answer).toContain('answer based on the context');
    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.keyFacts.length).toBeGreaterThan(0);
    expect(result.summary).toBeDefined();
    expect(result.usage.totalTokens).toBeGreaterThan(0);
  });

  it('should handle RAG query without embeddings (stub mode)', async () => {
    vi.doMock('../config', () => ({
      config: {
        openaiApiKey: null,
      },
    }));

    const { generateEmbedding } = await import('../services/embedding.service');
    vi.mocked(generateEmbedding).mockResolvedValue(null);

    const { prisma } = await import('../db');
    vi.mocked(prisma.documentChunk.findMany).mockResolvedValue([
      {
        id: 'chunk-1',
        content: 'Test content without embeddings',
        documentId: 'doc-1',
        chunkIndex: 0,
        userId: 'user-1',
        tokenCount: 10,
        embedding: null,
        createdAt: new Date(),
      },
    ]);

    vi.mocked(prisma.document.findMany).mockResolvedValue([
      {
        id: 'doc-1',
        originalName: 'test.pdf',
      } as any,
    ]);

    const { performRAGQuery: ragQuery } = await import('../services/rag.service');

    const result = await ragQuery('test question', 'collection-1', 'user-1');

    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.citations[0].relevanceScore).toBe(0);
  });

  it('should handle OpenAI API errors', async () => {
    vi.doMock('../config', () => ({
      config: {
        openaiApiKey: 'test-key',
      },
    }));

    const mockEmbedding = new Array(1536).fill(0.1);

    const { generateEmbedding } = await import('../services/embedding.service');
    vi.mocked(generateEmbedding).mockResolvedValue(mockEmbedding);

    const { prisma } = await import('../db');
    vi.mocked(prisma.$queryRaw).mockResolvedValue([
      {
        id: 'chunk-1',
        content: 'Test content',
        documentId: 'doc-1',
        chunkIndex: 0,
        score: 0.9,
      },
    ]);

    vi.mocked(prisma.document.findMany).mockResolvedValue([
      {
        id: 'doc-1',
        originalName: 'test.pdf',
      } as any,
    ]);

    const OpenAI = (await import('openai')).default;
    // @ts-ignore
    OpenAI.mockImplementation(() => ({
      chat: {
        completions: {
          create: vi.fn().mockRejectedValue(new Error('OpenAI API Error')),
        },
      },
      embeddings: {
        create: vi.fn().mockResolvedValue({
          data: [{ embedding: mockEmbedding }],
        }),
      },
    }));

    const { performRAGQuery: ragQuery } = await import('../services/rag.service');

    await expect(ragQuery('test question', 'collection-1', 'user-1')).rejects.toThrow('OpenAI API Error');
  });
});
