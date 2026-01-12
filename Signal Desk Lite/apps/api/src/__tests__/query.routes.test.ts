import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { prisma } from '../db';
import * as ragService from '../services/rag.service';

// Mock dependencies
vi.mock('../db', () => ({
    prisma: {
        collection: {
            findFirst: vi.fn(),
        },
        queryLog: {
            create: vi.fn(),
        },
    },
}));

vi.mock('../services/rag.service', () => ({
    performRAGQuery: vi.fn(),
}));

vi.mock('../middleware/auth', () => ({
    authMiddleware: async (request: any, _reply: any) => {
        request.user = { userId: 'test-user-id', email: 'test@example.com' };
    },
}));

describe('Query Routes', () => {
    let fastify: FastifyInstance;

    beforeEach(async () => {
        fastify = Fastify();
        await fastify.register(import('@fastify/cookie'));
        const { queryRoutes } = await import('../routes/query.routes');
        await fastify.register(queryRoutes);
        vi.clearAllMocks();
    });

    afterEach(async () => {
        await fastify.close();
    });

    describe('POST /collections/:id/query', () => {
        it('should process RAG query successfully', async () => {
            const collectionId = 'cljx4q8f20000k7j3h5t6n3c3';
            const mockResult = {
                answer: 'Test answer',
                citations: [{ chunkId: '1', documentId: 'doc-1', relevanceScore: 0.9 }],
                keyFacts: [{ fact: 'Fact 1', source: 'chunk-1' }],
                summary: 'Test summary',
                usage: { promptTokens: 10, completionTokens: 20, totalTokens: 30 },
            };

            vi.mocked(prisma.collection.findFirst).mockResolvedValue({ id: collectionId } as any);
            vi.mocked(ragService.performRAGQuery).mockResolvedValue(mockResult as any);
            vi.mocked(prisma.queryLog.create).mockResolvedValue({} as any);

            const response = await fastify.inject({
                method: 'POST',
                url: `/collections/${collectionId}/query`,
                payload: {
                    question: 'Test question',
                },
            });

            expect(response.statusCode).toBe(200);
            const body = JSON.parse(response.body);
            expect(body.data.answer).toBe('Test answer');
            expect(prisma.queryLog.create).toHaveBeenCalled();
        });

        it('should return 404 if collection not found', async () => {
            const collectionId = 'cljx4q8f20000k7j3h5t6n3c3';
            vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

            const response = await fastify.inject({
                method: 'POST',
                url: `/collections/${collectionId}/query`,
                payload: {
                    question: 'Test question',
                },
            });

            expect(response.statusCode).toBe(404);
        });

        it('should return 400 for invalid input', async () => {
            const collectionId = 'cljx4q8f20000k7j3h5t6n3c3';
            vi.mocked(prisma.collection.findFirst).mockResolvedValue({ id: collectionId } as any);

            const response = await fastify.inject({
                method: 'POST',
                url: `/collections/${collectionId}/query`,
                payload: {
                    // Missing question
                },
            });

            expect(response.statusCode).toBe(400);
        });
    });
});
