import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { prisma } from '../db';
import * as queueJob from '../jobs/queue';
import { CONFIG } from '@signaldesk/shared';

// Mock dependencies
vi.mock('../db', () => ({
    prisma: {
        collection: {
            findFirst: vi.fn(),
        },
        document: {
            findMany: vi.fn(),
            findFirst: vi.fn(),
            create: vi.fn(),
            delete: vi.fn(),
        },
    },
}));

vi.mock('../jobs/queue', () => ({
    enqueueDocumentProcessing: vi.fn(),
}));

vi.mock('../middleware/auth', () => ({
    authMiddleware: async (request: any, _reply: any) => {
        request.user = { userId: 'test-user-id', email: 'test@example.com' };
    },
}));

vi.mock('fs/promises', () => ({
    default: {
        mkdir: vi.fn(),
        rm: vi.fn(),
        createWriteStream: vi.fn(),
    },
}));

vi.mock('stream/promises', () => ({
    pipeline: vi.fn(),
}));

// Mock multipart plugin
// We need to mock the request.file() behavior for upload tests
// But fastify-multipart logic is tricky to mock perfectly without actually using form-data.
// Since we inject, we should use form-data or mock the parser.
// However, fastify-multipart decorates request.
// For unit testing routes, maybe we can mock FastifyRequest.file method?
// Actually simpler: just rely on fastify-multipart being registered and send multipart data in inject.

describe('Documents Routes', () => {
    let fastify: FastifyInstance;

    beforeEach(async () => {
        fastify = Fastify();
        await fastify.register(import('@fastify/multipart'));
        await fastify.register(import('@fastify/cookie'));
        const { documentsRoutes } = await import('../routes/documents.routes');
        await fastify.register(documentsRoutes);
        vi.clearAllMocks();
    });

    afterEach(async () => {
        await fastify.close();
    });

    describe('GET /collections/:collectionId/documents', () => {
        it('should list documents for a collection', async () => {
            const collectionId = 'cljx4q8f10000k7j3h5t6m1a1';
            vi.mocked(prisma.collection.findFirst).mockResolvedValue({ id: collectionId } as any);
            vi.mocked(prisma.document.findMany).mockResolvedValue([
                {
                    id: 'cljx4q8f10001k7j3h5t6m2b2',
                    filename: 'test.pdf',
                    originalName: 'test.pdf',
                    mimeType: 'application/pdf',
                    sizeBytes: 1024,
                    status: 'COMPLETED',
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    collectionId,
                    _count: { chunks: 5 },
                } as any,
            ]);

            const response = await fastify.inject({
                method: 'GET',
                url: `/collections/${collectionId}/documents`,
            });

            expect(response.statusCode).toBe(200);
            const body = JSON.parse(response.body);
            expect(body.data.documents).toHaveLength(1);
            expect(body.data.documents[0].id).toBe('cljx4q8f10001k7j3h5t6m2b2');
            expect(body.data.documents[0].chunkCount).toBe(5);
        });

        it('should return 404 if collection not found', async () => {
            vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

            const response = await fastify.inject({
                method: 'GET',
                url: '/collections/cljx4q8f10000k7j3h5t6m1a1/documents',
            });

            expect(response.statusCode).toBe(404);
        });
    });

    describe('GET /documents/:id', () => {
        it('should get document details', async () => {
            const docId = 'cljx4q8f10001k7j3h5t6m2b2';
            vi.mocked(prisma.document.findFirst).mockResolvedValue({
                id: docId,
                filename: 'test.pdf',
                chunks: [],
            } as any);

            const response = await fastify.inject({
                method: 'GET',
                url: `/documents/${docId}`,
            });

            expect(response.statusCode).toBe(200);
            const body = JSON.parse(response.body);
            expect(body.data.document.id).toBe(docId);
        });

        it('should return 404 if document not found', async () => {
            vi.mocked(prisma.document.findFirst).mockResolvedValue(null);

            const response = await fastify.inject({
                method: 'GET',
                url: '/documents/cljx4q8f10001k7j3h5t6m2b2',
            });

            expect(response.statusCode).toBe(404);
        });
    });

    describe('DELETE /documents/:id', () => {
        it('should delete document', async () => {
            const docId = 'cljx4q8f10001k7j3h5t6m2b2';
            vi.mocked(prisma.document.findFirst).mockResolvedValue({
                id: docId,
            } as any);
            vi.mocked(prisma.document.delete).mockResolvedValue({} as any);

            const response = await fastify.inject({
                method: 'DELETE',
                url: `/documents/${docId}`,
            });

            expect(response.statusCode).toBe(200);
            expect(prisma.document.delete).toHaveBeenCalledWith({ where: { id: docId } });
        });

        it('should return 404 if document not found', async () => {
            vi.mocked(prisma.document.findFirst).mockResolvedValue(null);

            const response = await fastify.inject({
                method: 'DELETE',
                url: '/documents/cljx4q8f10001k7j3h5t6m2b2',
            });

            expect(response.statusCode).toBe(404);
        });
    });
});
