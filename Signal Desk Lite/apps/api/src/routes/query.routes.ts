import { FastifyInstance } from 'fastify';
import { querySchema } from '@signaldesk/shared';
import { prisma } from '../db';
import { authMiddleware } from '../middleware/auth';
import { performRAGQuery } from '../services/rag.service';
import { uuidParamSchema } from '../utils/validation';

export async function queryRoutes(fastify: FastifyInstance) {
  fastify.addHook('preHandler', authMiddleware);

  fastify.post('/collections/:id/query', async (request, reply) => {
    const startTime = Date.now();

    try {
      const { id } = uuidParamSchema.parse(request.params);

      const collection = await prisma.collection.findFirst({
      where: {
        id,
        userId: request.user!.userId,
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

      const body = querySchema.parse(request.body);

      const result = await performRAGQuery(body.question, id, request.user!.userId);

      const latencyMs = Date.now() - startTime;

      await prisma.queryLog.create({
        data: {
          userId: request.user!.userId,
          collectionId: id,
          question: body.question,
          answer: result.answer,
          citationChunks: result.citations.map((c) => c.chunkId),
          keyFacts: result.keyFacts,
          summary: result.summary,
          promptTokens: result.usage.promptTokens,
          completionTokens: result.usage.completionTokens,
          totalTokens: result.usage.totalTokens,
          latencyMs,
        },
      });

      return reply.send({
        data: result,
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
}
