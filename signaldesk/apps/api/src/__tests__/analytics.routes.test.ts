import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { prisma } from '../db';
import { CONFIG } from '@signaldesk/shared';

// Mock dependencies
vi.mock('../db', () => ({
    prisma: {
        document: {
            count: vi.fn(),
        },
        documentChunk: {
            count: vi.fn(),
        },
        queryLog: {
            findMany: vi.fn(),
        },
        collection: {
            findFirst: vi.fn(),
        },
    },
}));

vi.mock('../middleware/auth', () => ({
    authMiddleware: async (request: any, _reply: any) => {
        request.user = { userId: 'test-user-id', email: 'test@example.com' };
    },
}));

describe('Analytics Routes', () => {
    let fastify: FastifyInstance;

    beforeEach(async () => {
        fastify = Fastify();
        await fastify.register(import('@fastify/cookie'));
        const { analyticsRoutes } = await import('../routes/analytics.routes');
        await fastify.register(analyticsRoutes);
        vi.clearAllMocks();
    });

    afterEach(async () => {
        await fastify.close();
    });

    describe('GET /analytics', () => {
        it('should return aggregated analytics for user', async () => {
            vi.mocked(prisma.document.count).mockResolvedValue(10);
            vi.mocked(prisma.documentChunk.count).mockResolvedValue(100);
            vi.mocked(prisma.queryLog.findMany).mockResolvedValue([
                {
                    promptTokens: 100,
                    completionTokens: 50,
                    totalTokens: 150,
                },
                {
                    promptTokens: 200,
                    completionTokens: 100,
                    totalTokens: 300,
                },
            ] as any);

            const response = await fastify.inject({
                method: 'GET',
                url: '/analytics',
            });

            expect(response.statusCode).toBe(200);
            const body = JSON.parse(response.body);
            expect(body.data.analytics.documentCount).toBe(10);
            expect(body.data.analytics.chunkCount).toBe(100);
            expect(body.data.analytics.queryCount).toBe(2);
            expect(body.data.analytics.tokenUsage.totalTokens).toBe(450);
            expect(body.data.analytics.estimatedCost.totalCost).toBeGreaterThan(0);
        });
    });

    describe('GET /collections/:id/analytics', () => {
        it('should return collection analytics', async () => {
            const collectionId = 'cljx4q8f00000k7j3h5t6l9vn';
            vi.mocked(prisma.collection.findFirst).mockResolvedValue({ id: collectionId } as any);
            vi.mocked(prisma.document.count).mockResolvedValue(5);
            vi.mocked(prisma.documentChunk.count).mockResolvedValue(50);
            vi.mocked(prisma.queryLog.findMany).mockResolvedValue([
                {
                    promptTokens: 50,
                    completionTokens: 20,
                    totalTokens: 70,
                    latencyMs: 100,
                },
            ] as any);

            const response = await fastify.inject({
                method: 'GET',
                url: `/collections/${collectionId}/analytics`,
            });

            expect(response.statusCode).toBe(200);
            const body = JSON.parse(response.body);
            expect(body.data.analytics.documentCount).toBe(5);
            expect(body.data.analytics.avgLatencyMs).toBe(100);
        });

        it('should return 404 if collection not found', async () => {
            vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

            const response = await fastify.inject({
                method: 'GET',
                url: '/collections/cljx4q8f00000k7j3h5t6l9vn/analytics',
            });

            expect(response.statusCode).toBe(404);
        });
    });
});
