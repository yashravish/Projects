import { FastifyInstance } from 'fastify';
import { createCollectionSchema, updateCollectionSchema } from '@signaldesk/shared';
import { prisma } from '../db';
import { authMiddleware } from '../middleware/auth';
import { uuidParamSchema } from '../utils/validation';

export async function collectionsRoutes(fastify: FastifyInstance) {
  fastify.addHook('preHandler', authMiddleware);

  fastify.get('/', async (request, reply) => {
    const collections = await prisma.collection.findMany({
      where: { userId: request.user!.userId },
      orderBy: { createdAt: 'desc' },
      include: {
        _count: {
          select: {
            documents: true,
          },
        },
      },
    });

    return reply.send({
      data: {
        collections: collections.map((collection: typeof collections[0]) => ({
          id: collection.id,
          name: collection.name,
          description: collection.description,
          createdAt: collection.createdAt,
          updatedAt: collection.updatedAt,
          documentCount: collection._count.documents,
        })),
      },
    });
  });

  fastify.post('/', async (request, reply) => {
    try {
      const body = createCollectionSchema.parse(request.body);

      const collection = await prisma.collection.create({
        data: {
          userId: request.user!.userId,
          name: body.name,
          description: body.description,
        },
      });

      return reply.status(201).send({
        data: {
          collection: {
            id: collection.id,
            name: collection.name,
            description: collection.description,
            createdAt: collection.createdAt,
            updatedAt: collection.updatedAt,
          },
        },
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid input',
            details: error,
          },
        });
      }
      throw error;
    }
  });

  fastify.get('/:id', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);

      const collection = await prisma.collection.findFirst({
      where: {
        id,
        userId: request.user!.userId,
      },
      include: {
        _count: {
          select: {
            documents: true,
            queryLogs: true,
          },
        },
      },
    });

    if (!collection) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Collection not found',
        },
      });
    }

    return reply.send({
      data: {
        collection: {
          id: collection.id,
          name: collection.name,
          description: collection.description,
          createdAt: collection.createdAt,
          updatedAt: collection.updatedAt,
          documentCount: collection._count.documents,
          queryCount: collection._count.queryLogs,
        },
      },
    });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid path parameter',
            details: error,
          },
        });
      }
      throw error;
    }
  });

  fastify.patch('/:id', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);
      const body = updateCollectionSchema.parse(request.body);

      const existingCollection = await prisma.collection.findFirst({
        where: {
          id,
          userId: request.user!.userId,
        },
      });

      if (!existingCollection) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'Collection not found',
          },
        });
      }

      const collection = await prisma.collection.update({
        where: { id },
        data: body,
      });

      return reply.send({
        data: {
          collection: {
            id: collection.id,
            name: collection.name,
            description: collection.description,
            createdAt: collection.createdAt,
            updatedAt: collection.updatedAt,
          },
        },
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid input',
            details: error,
          },
        });
      }
      throw error;
    }
  });

  fastify.delete('/:id', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);

      const existingCollection = await prisma.collection.findFirst({
      where: {
        id,
        userId: request.user!.userId,
      },
    });

    if (!existingCollection) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Collection not found',
        },
      });
    }

    await prisma.collection.delete({
      where: { id },
    });

    return reply.send({
      data: {
        message: 'Collection deleted successfully',
      },
    });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid path parameter',
            details: error,
          },
        });
      }
      throw error;
    }
  });
}
