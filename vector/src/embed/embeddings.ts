import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import type { VDiagramConfig } from '../config.js';

export type EmbedOptions = {
  batchSize?: number;
  concurrency?: number;
  allowNetwork?: boolean;
};

export type Embedder = {
  embedTexts: (texts: string[], options?: EmbedOptions) => Promise<Float32Array[]>;
};

export async function createEmbedder(cfg: VDiagramConfig, allowNetwork: boolean): Promise<Embedder> {
  const cacheDir = cfg.modelPath.replace(/^~/, os.homedir());
  process.env.TRANSFORMERS_CACHE = cacheDir;
  if (!allowNetwork) process.env.TRANSFORMERS_OFFLINE = '1'; else delete process.env.TRANSFORMERS_OFFLINE;
  const { pipeline } = await import('@xenova/transformers');
  let pipe: any;
  try {
    pipe = await pipeline('feature-extraction', `Xenova/${cfg.embeddingModel}`);
  } catch (e) {
    if (!allowNetwork) {
      throw new Error(`Embedding model not found at ${cacheDir}. Run: vdiagram models pull --allow-network`);
    }
    throw e;
  }

  async function embedBatch(batch: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (const text of batch) {
      const result: any = await pipe(text, { pooling: 'mean', normalize: true });
      // result.data is a TypedArray
      const arr = new Float32Array(result.data);
      out.push(arr);
    }
    return out;
  }

  return {
    async embedTexts(texts: string[], options?: EmbedOptions): Promise<Float32Array[]> {
      const batchSize = options?.batchSize ?? 128;
      const out: Float32Array[] = [];
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        const emb = await embedBatch(batch);
        out.push(...emb);
      }
      return out;
    },
  };
}


