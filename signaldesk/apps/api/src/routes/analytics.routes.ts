import { FastifyInstance } from 'fastify';
import { CONFIG } from '@signaldesk/shared';
import { prisma } from '../db';
import { authMiddleware } from '../middleware/auth';
import { uuidParamSchema } from '../utils/validation';

export async function analyticsRoutes(fastify: FastifyInstance) {
  fastify.addHook('preHandler', authMiddleware);

  fastify.get('/analytics', async (request, reply) => {
    const userId = request.user!.userId;

    const [documentCount, chunkCount, queryLogs] = await Promise.all([
      prisma.document.count({
        where: { userId },
      }),
      prisma.documentChunk.count({
        where: { userId },
      }),
      prisma.queryLog.findMany({
        where: { userId },
        select: {
          promptTokens: true,
          completionTokens: true,
          totalTokens: true,
        },
      }),
    ]);

    const queryCount = queryLogs.length;
    const totalPromptTokens = queryLogs.reduce((sum: number, log) => sum + log.promptTokens, 0);
    const totalCompletionTokens = queryLogs.reduce((sum: number, log) => sum + log.completionTokens, 0);
    const totalTokens = queryLogs.reduce((sum: number, log) => sum + log.totalTokens, 0);

    const embeddingCost =
      (totalPromptTokens / 1_000_000) * CONFIG.EMBEDDING_COST_PER_1M_TOKENS;
    const chatInputCost =
      (totalPromptTokens / 1_000_000) * CONFIG.CHAT_INPUT_COST_PER_1M_TOKENS;
    const chatOutputCost =
      (totalCompletionTokens / 1_000_000) * CONFIG.CHAT_OUTPUT_COST_PER_1M_TOKENS;
    const totalCost = embeddingCost + chatInputCost + chatOutputCost;

    return reply.send({
      data: {
        analytics: {
          documentCount,
          chunkCount,
          queryCount,
          tokenUsage: {
            promptTokens: totalPromptTokens,
            completionTokens: totalCompletionTokens,
            totalTokens,
          },
          estimatedCost: {
            embeddingCost: parseFloat(embeddingCost.toFixed(4)),
            chatInputCost: parseFloat(chatInputCost.toFixed(4)),
            chatOutputCost: parseFloat(chatOutputCost.toFixed(4)),
            totalCost: parseFloat(totalCost.toFixed(4)),
          },
        },
      },
    });
  });

  fastify.get('/collections/:id/analytics', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);
      const userId = request.user!.userId;

      const collection = await prisma.collection.findFirst({
      where: {
        id,
        userId,
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

    const [documentCount, chunkCount, queryLogs] = await Promise.all([
      prisma.document.count({
        where: {
          collectionId: id,
          userId,
        },
      }),
      prisma.documentChunk.count({
        where: {
          document: {
            collectionId: id,
          },
          userId,
        },
      }),
      prisma.queryLog.findMany({
        where: {
          collectionId: id,
          userId,
        },
        select: {
          promptTokens: true,
          completionTokens: true,
          totalTokens: true,
          latencyMs: true,
        },
      }),
    ]);

    const queryCount = queryLogs.length;
    const totalPromptTokens = queryLogs.reduce((sum: number, log) => sum + log.promptTokens, 0);
    const totalCompletionTokens = queryLogs.reduce((sum: number, log) => sum + log.completionTokens, 0);
    const totalTokens = queryLogs.reduce((sum: number, log) => sum + log.totalTokens, 0);
    const avgLatencyMs =
      queryLogs.length > 0
        ? queryLogs.reduce((sum: number, log) => sum + log.latencyMs, 0) / queryLogs.length
        : 0;

    const embeddingCost =
      (totalPromptTokens / 1_000_000) * CONFIG.EMBEDDING_COST_PER_1M_TOKENS;
    const chatInputCost =
      (totalPromptTokens / 1_000_000) * CONFIG.CHAT_INPUT_COST_PER_1M_TOKENS;
    const chatOutputCost =
      (totalCompletionTokens / 1_000_000) * CONFIG.CHAT_OUTPUT_COST_PER_1M_TOKENS;
    const totalCost = embeddingCost + chatInputCost + chatOutputCost;

    return reply.send({
      data: {
        analytics: {
          documentCount,
          chunkCount,
          queryCount,
          avgLatencyMs: parseFloat(avgLatencyMs.toFixed(2)),
          tokenUsage: {
            promptTokens: totalPromptTokens,
            completionTokens: totalCompletionTokens,
            totalTokens,
          },
          estimatedCost: {
            embeddingCost: parseFloat(embeddingCost.toFixed(4)),
            chatInputCost: parseFloat(chatInputCost.toFixed(4)),
            chatOutputCost: parseFloat(chatOutputCost.toFixed(4)),
            totalCost: parseFloat(totalCost.toFixed(4)),
          },
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
}
