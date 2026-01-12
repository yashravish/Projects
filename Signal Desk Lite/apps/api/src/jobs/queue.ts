import { Queue } from 'bullmq';
import { config } from '../config';

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
console.log('📮 Queue connecting to Redis:', connection.host, 'TLS:', !!connection.tls);

export const documentQueue = new Queue('document-processing', { connection });

export interface ProcessDocumentJob {
  documentId: string;
}

export async function enqueueDocumentProcessing(documentId: string) {
  await documentQueue.add('process-document', { documentId } as ProcessDocumentJob);
}

