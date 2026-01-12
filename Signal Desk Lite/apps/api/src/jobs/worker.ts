import { Worker } from 'bullmq';
import path from 'path';
import { prisma } from '../db';
import { config } from '../config';
import { extractTextFromFile } from '../services/extraction.service';
import { chunkText, estimateTokenCount } from '../services/chunking.service';
import { generateEmbeddings } from '../services/embedding.service';
import { ProcessDocumentJob } from './queue';

// Parse Redis URL and configure connection for Upstash TLS
function parseRedisConnection() {
  const url = new URL(config.redisUrl);
  const isUpstash = url.hostname.includes('upstash.io');

  return {
    host: url.hostname,
    port: parseInt(url.port) || 6379,
    password: url.password || undefined,
    username: url.username || 'default',
    maxRetriesPerRequest: null, // Required for BullMQ workers
    connectTimeout: 10000, // 10 seconds
    keepAlive: 30000, // 30 seconds
    family: 4, // IPv4
    ...(isUpstash && {
      tls: { rejectUnauthorized: false },
      enableTLSForSentinelMode: false,
    }),
  };
}

const connection = parseRedisConnection();

export const documentWorker = new Worker(
  'document-processing',
  async (job) => {
    const { documentId } = job.data as ProcessDocumentJob;

    try {
      const document = await prisma.document.findUnique({
        where: { id: documentId },
      });

      if (!document) {
        throw new Error(`Document ${documentId} not found`);
      }

      await prisma.document.update({
        where: { id: documentId },
        data: { status: 'PROCESSING' },
      });

      const storagePath = path.join(
        __dirname,
        '../../storage',
        document.userId,
        documentId,
        document.filename
      );

      const text = await extractTextFromFile(storagePath, document.mimeType);

      const chunks = chunkText(text);

      const embeddings = await generateEmbeddings(chunks);

      const chunkRecords = chunks.map((content, index) => ({
        id: `${documentId}-chunk-${index}`,
        documentId,
        userId: document.userId,
        chunkIndex: index,
        content,
        tokenCount: estimateTokenCount(content),
        embedding: embeddings[index] ? `[${embeddings[index]!.join(',')}]` : null,
      }));

      await prisma.$transaction(
        chunkRecords.map((chunk) =>
          prisma.$executeRaw`
            INSERT INTO "DocumentChunk" (id, "documentId", "userId", "chunkIndex", content, "tokenCount", embedding, "createdAt")
            VALUES (
              ${chunk.id},
              ${chunk.documentId},
              ${chunk.userId},
              ${chunk.chunkIndex},
              ${chunk.content},
              ${chunk.tokenCount},
              ${chunk.embedding ? chunk.embedding : null}::vector,
              NOW()
            )
          `
        )
      );

      await prisma.document.update({
        where: { id: documentId },
        data: { status: 'COMPLETED' },
      });
    } catch (error) {
      console.error(`Error processing document ${documentId}:`, error);

      await prisma.document.update({
        where: { id: documentId },
        data: {
          status: 'FAILED',
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
        },
      });

      throw error;
    }
  },
  {
    connection,
    concurrency: 2,
  }
);

documentWorker.on('completed', (job) => {
  console.log(`✅ Document processing job ${job.id} completed`);
});

documentWorker.on('failed', (job, err) => {
  console.error(`❌ Document processing job ${job?.id} failed:`, err);
});

documentWorker.on('error', (err) => {
  console.error('🔴 Worker connection error:', err);
});

documentWorker.on('ready', () => {
  console.log('🔧 Document worker is ready and listening for jobs');
});

documentWorker.on('active', (job) => {
  console.log(`🔄 Processing job ${job.id} for document:`, job.data.documentId);
});

documentWorker.on('stalled', (jobId) => {
  console.warn(`⚠️ Job ${jobId} stalled`);
});

console.log('📋 Document worker started, connecting to Redis...');
