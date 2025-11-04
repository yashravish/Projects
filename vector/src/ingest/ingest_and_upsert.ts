import { runIngest } from './pipeline.js';
import type { VDiagramConfig } from '../config.js';
import { createEmbedder } from '../embed/embeddings.js';
import { openVectorStore } from '../store/index.js';
import type { ChunkRecord } from '../store/types.js';

type Logger = { info: (m: string) => void; debug: (m: string) => void; warn: (m: string) => void; error: (m: string) => void };

export async function ingestEmbedUpsert(
  inputPaths: string[],
  cfg: VDiagramConfig,
  logger: Logger,
  opts: {
    includeGlobs?: string[];
    ignoreGlobs?: string[];
    maxFileSizeBytes?: number;
    targetChunkSize?: number;
    overlap?: number;
    maxChunkSize?: number;
    tag?: string;
    reset?: boolean;
    batchSize?: number;
    concurrency?: number;
    allowNetwork?: boolean;
  }
) {
  const { stats, chunks } = await runIngest(inputPaths, cfg, logger, opts);
  if (chunks.length === 0) {
    logger.info('No new or changed chunks to embed.');
    return { stats, upserted: 0 };
  }
  const embedder = await createEmbedder(cfg, !!opts.allowNetwork);
  logger.info(`Embedding ${chunks.length} chunks...`);
  const texts = chunks.map((c) => c.content);
  const vectors = await embedder.embedTexts(texts, { batchSize: opts.batchSize, concurrency: opts.concurrency });
  const now = Date.now();
  const records: ChunkRecord[] = chunks.map((c, i) => ({
    id: c.id,
    path: c.path,
    startLine: c.startLine,
    endLine: c.endLine,
    content: c.content,
    embedding: vectors[i],
    lang: c.lang,
    tags: opts.tag ? [opts.tag] : undefined,
    hash: undefined,
    ingestedAt: now,
  }));
  const store = await openVectorStore(cfg.dbPath);
  await store.upsertChunks(records, opts.tag);
  logger.info(`Upserted ${records.length} chunk vectors.`);
  return { stats, upserted: records.length };
}


