import { FastifyInstance } from 'fastify';
import fs from 'fs/promises';
import { createWriteStream } from 'fs';
import path from 'path';
import { pipeline } from 'stream/promises';
import { CONFIG } from '@signaldesk/shared';
import { prisma } from '../db';
import { authMiddleware } from '../middleware/auth';
import { enqueueDocumentProcessing } from '../jobs/queue';
import { sanitizeFilename } from '../utils/file';
import { collectionIdParamSchema, uuidParamSchema } from '../utils/validation';

export async function documentsRoutes(fastify: FastifyInstance) {
  fastify.addHook('preHandler', authMiddleware);

  fastify.get('/collections/:collectionId/documents', async (request, reply) => {
    try {
      const { collectionId } = collectionIdParamSchema.parse(request.params);

      const collection = await prisma.collection.findFirst({
        where: {
          id: collectionId,
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

      const documents = await prisma.document.findMany({
        where: {
          collectionId,
          userId: request.user!.userId,
        },
        orderBy: { createdAt: 'desc' },
        include: {
          _count: {
            select: {
              chunks: true,
            },
          },
        },
      });

      return reply.send({
        data: {
          documents: documents.map((doc) => ({
            id: doc.id,
            filename: doc.filename,
            originalName: doc.originalName,
            mimeType: doc.mimeType,
            sizeBytes: doc.sizeBytes,
            status: doc.status,
            errorMessage: doc.errorMessage,
            chunkCount: doc._count.chunks,
            createdAt: doc.createdAt,
            updatedAt: doc.updatedAt,
          })),
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

  fastify.post('/collections/:collectionId/documents', async (request, reply) => {
    try {
      const { collectionId } = collectionIdParamSchema.parse(request.params);

      const collection = await prisma.collection.findFirst({
        where: {
          id: collectionId,
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
      const data = await request.file();

      if (!data) {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'No file uploaded',
          },
        });
      }

      if (data.file.bytesRead > CONFIG.MAX_FILE_SIZE_BYTES) {
        return reply.status(413).send({
          error: {
            code: 'PAYLOAD_TOO_LARGE',
            message: `File size exceeds ${CONFIG.MAX_FILE_SIZE_BYTES} bytes`,
          },
        });
      }

      if (!CONFIG.ALLOWED_MIME_TYPES.includes(data.mimetype as never)) {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: `Invalid file type. Allowed types: ${CONFIG.ALLOWED_MIME_TYPES.join(', ')}`,
          },
        });
      }

      const extension = path.extname(data.filename);
      if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension as never)) {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: `Invalid file extension. Allowed extensions: ${CONFIG.ALLOWED_EXTENSIONS.join(', ')}`,
          },
        });
      }

      const sanitizedFilename = sanitizeFilename(data.filename);

      const document = await prisma.document.create({
        data: {
          collectionId,
          userId: request.user!.userId,
          filename: sanitizedFilename,
          originalName: data.filename,
          mimeType: data.mimetype,
          sizeBytes: data.file.bytesRead,
        },
      });

      const storageDir = path.join(
        __dirname,
        '../../storage',
        request.user!.userId,
        document.id
      );
      await fs.mkdir(storageDir, { recursive: true });

      const filePath = path.join(storageDir, sanitizedFilename);
      await pipeline(data.file, createWriteStream(filePath));

      await enqueueDocumentProcessing(document.id);

      return reply.status(202).send({
        data: {
          document: {
            id: document.id,
            filename: document.filename,
            originalName: document.originalName,
            mimeType: document.mimeType,
            sizeBytes: document.sizeBytes,
            status: document.status,
            createdAt: document.createdAt,
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
      console.error('Error uploading document:', error);
      throw error;
    }
  });

  fastify.get('/documents/:id', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);

      const document = await prisma.document.findFirst({
        where: {
          id,
          userId: request.user!.userId,
        },
        include: {
          chunks: {
            orderBy: { chunkIndex: 'asc' },
            select: {
              id: true,
              chunkIndex: true,
              content: true,
              tokenCount: true,
            },
          },
        },
      });

      if (!document) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'Document not found',
          },
        });
      }

      return reply.send({
        data: {
          document: {
            id: document.id,
            filename: document.filename,
            originalName: document.originalName,
            mimeType: document.mimeType,
            sizeBytes: document.sizeBytes,
            status: document.status,
            errorMessage: document.errorMessage,
            createdAt: document.createdAt,
            updatedAt: document.updatedAt,
            chunks: document.chunks,
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

  fastify.delete('/documents/:id', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);

      const document = await prisma.document.findFirst({
        where: {
          id,
          userId: request.user!.userId,
        },
      });

      if (!document) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'Document not found',
          },
        });
      }

      const storageDir = path.join(__dirname, '../../storage', request.user!.userId, document.id);

      try {
        await fs.rm(storageDir, { recursive: true, force: true });
      } catch (error) {
        console.warn('Failed to delete document files:', error);
      }

      await prisma.document.delete({
        where: { id },
      });

      return reply.send({
        data: {
          message: 'Document deleted successfully',
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

  // Reprocess a document that is stuck in PENDING or FAILED status
  fastify.post('/documents/:id/reprocess', async (request, reply) => {
    try {
      const { id } = uuidParamSchema.parse(request.params);

      const document = await prisma.document.findFirst({
        where: {
          id,
          userId: request.user!.userId,
        },
      });

      if (!document) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'Document not found',
          },
        });
      }

      if (document.status !== 'PENDING' && document.status !== 'FAILED') {
        return reply.status(400).send({
          error: {
            code: 'INVALID_STATUS',
            message: 'Document can only be reprocessed if status is PENDING or FAILED',
          },
        });
      }

      // Reset status to PENDING and re-enqueue
      await prisma.document.update({
        where: { id },
        data: { status: 'PENDING', errorMessage: null },
      });

      await enqueueDocumentProcessing(document.id);

      return reply.send({
        data: {
          message: 'Document reprocessing queued',
          document: {
            id: document.id,
            status: 'PENDING',
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
