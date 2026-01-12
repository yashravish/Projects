import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { collectionsRoutes } from '../routes/collections.routes';
import { prisma } from '../db';

// Mock dependencies
vi.mock('../db', () => ({
  prisma: {
    collection: {
      findMany: vi.fn(),
      findFirst: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    },
  },
}));

vi.mock('../middleware/auth', () => ({
  authMiddleware: async (request: any, _reply: any) => {
    request.user = { userId: 'test-user-id', email: 'test@example.com' };
  },
}));

describe('Collections Routes', () => {
  let fastify: FastifyInstance;

  beforeEach(async () => {
    fastify = Fastify();
    await fastify.register(collectionsRoutes, { prefix: '/collections' });
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await fastify.close();
  });

  describe('GET /collections', () => {
    it('should return list of collections', async () => {
      const mockCollections = [
        {
          id: 'collection-1',
          name: 'Test Collection 1',
          description: 'Description 1',
          userId: 'test-user-id',
          createdAt: new Date(),
          updatedAt: new Date(),
          _count: {
            documents: 5,
          },
        },
        {
          id: 'collection-2',
          name: 'Test Collection 2',
          description: 'Description 2',
          userId: 'test-user-id',
          createdAt: new Date(),
          updatedAt: new Date(),
          _count: {
            documents: 3,
          },
        },
      ];

      vi.mocked(prisma.collection.findMany).mockResolvedValue(mockCollections);

      const response = await fastify.inject({
        method: 'GET',
        url: '/collections',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.collections).toHaveLength(2);
      expect(body.data.collections[0].documentCount).toBe(5);
    });

    it('should return empty array when no collections exist', async () => {
      vi.mocked(prisma.collection.findMany).mockResolvedValue([]);

      const response = await fastify.inject({
        method: 'GET',
        url: '/collections',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.collections).toHaveLength(0);
    });
  });

  describe('POST /collections', () => {
    it('should create a new collection', async () => {
      const mockCollection = {
        id: 'new-collection-id',
        name: 'New Collection',
        description: 'New Description',
        userId: 'test-user-id',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      vi.mocked(prisma.collection.create).mockResolvedValue(mockCollection);

      const response = await fastify.inject({
        method: 'POST',
        url: '/collections',
        payload: {
          name: 'New Collection',
          description: 'New Description',
        },
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.data.collection.name).toBe('New Collection');
    });

    it('should return 400 for invalid input', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/collections',
        payload: {
          name: '', // Invalid: empty name
        },
      });

      expect(response.statusCode).toBe(400);
    });
  });

  describe('GET /collections/:id', () => {
    it('should return collection by id', async () => {
      const collectionId = 'cljx4q8f00000k7j3h5t6l9vn';
      const mockCollection = {
        id: collectionId,
        name: 'Test Collection',
        description: 'Test Description',
        userId: 'test-user-id',
        createdAt: new Date(),
        updatedAt: new Date(),
        _count: {
          documents: 10,
          queryLogs: 5,
        },
      };

      vi.mocked(prisma.collection.findFirst).mockResolvedValue(mockCollection);

      const response = await fastify.inject({
        method: 'GET',
        url: `/collections/${collectionId}`,
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.collection.name).toBe('Test Collection');
      expect(body.data.collection.documentCount).toBe(10);
      expect(body.data.collection.queryCount).toBe(5);
    });

    it('should return 404 if collection not found', async () => {
      const collectionId = 'cljx4q8f00001k7j3h5t6laaa';
      vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

      const response = await fastify.inject({
        method: 'GET',
        url: `/collections/${collectionId}`,
      });

      expect(response.statusCode).toBe(404);
    });
  });

  describe('PATCH /collections/:id', () => {
    it('should update collection', async () => {
      const collectionId = 'cljx4q8f00002k7j3h5t6lbbb';
      const existingCollection = {
        id: collectionId,
        name: 'Old Name',
        description: 'Old Description',
        userId: 'test-user-id',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      const updatedCollection = {
        ...existingCollection,
        name: 'Updated Name',
        description: 'Updated Description',
      };

      vi.mocked(prisma.collection.findFirst).mockResolvedValue(existingCollection);
      vi.mocked(prisma.collection.update).mockResolvedValue(updatedCollection);

      const response = await fastify.inject({
        method: 'PATCH',
        url: `/collections/${collectionId}`,
        payload: {
          name: 'Updated Name',
          description: 'Updated Description',
        },
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.collection.name).toBe('Updated Name');
    });

    it('should return 404 if collection not found', async () => {
      const collectionId = 'cljx4q8f00003k7j3h5t6lccc';
      vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

      const response = await fastify.inject({
        method: 'PATCH',
        url: `/collections/${collectionId}`,
        payload: {
          name: 'Updated Name',
        },
      });

      expect(response.statusCode).toBe(404);
    });
  });

  describe('DELETE /collections/:id', () => {
    it('should delete collection', async () => {
      const collectionId = 'cljx4q8f00004k7j3h5t6lddd';
      const existingCollection = {
        id: collectionId,
        name: 'Test Collection',
        description: 'Test Description',
        userId: 'test-user-id',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      vi.mocked(prisma.collection.findFirst).mockResolvedValue(existingCollection);
      vi.mocked(prisma.collection.delete).mockResolvedValue(existingCollection);

      const response = await fastify.inject({
        method: 'DELETE',
        url: `/collections/${collectionId}`,
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.message).toBe('Collection deleted successfully');
    });

    it('should return 404 if collection not found', async () => {
      const collectionId = 'cljx4q8f00005k7j3h5t6leee';
      vi.mocked(prisma.collection.findFirst).mockResolvedValue(null);

      const response = await fastify.inject({
        method: 'DELETE',
        url: `/collections/${collectionId}`,
      });

      expect(response.statusCode).toBe(404);
    });
  });
});
