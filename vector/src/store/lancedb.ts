import path from 'node:path';
import { expandHome } from '../config.js';
import type { VectorStore, ChunkRecord, QueryFilter, QueryResult } from './types.js';

export async function probeLanceDB(): Promise<boolean> {
  try {
    const mod = '@lancedb/lancedb';
    await import(mod);
    return true;
  } catch {
    return false;
  }
}

export class LanceDBStore implements VectorStore {
  private table: any;
  private dims: number | null = null;
  constructor(private dbPath: string) {}

  private async init(): Promise<void> {
    if (this.table) return;
    const mod = '@lancedb/lancedb';
    const lancedb: any = await import(mod);
    const abs = expandHome(this.dbPath);
    const uri = path.resolve(abs);
    const db = await lancedb.connect(uri);
    const tableName = 'chunks';
    const names = await db.tableNames();
    if (!names.includes(tableName)) {
      this.table = await db.createEmptyTable(tableName, {
        id: lancedb.UTF8,
        path: lancedb.UTF8,
        startLine: lancedb.Int32,
        endLine: lancedb.Int32,
        content: lancedb.UTF8,
        embedding: lancedb.Vector(lancedb.Float32, 384),
        lang: lancedb.UTF8,
        tags: lancedb.List(lancedb.UTF8),
        hash: lancedb.UTF8,
        ingestedAt: lancedb.Int64,
      });
    } else {
      this.table = await db.openTable(tableName);
    }
  }

  async upsertChunks(chunks: ChunkRecord[], tag?: string): Promise<void> {
    await this.init();
    if (chunks.length === 0) return;
    this.dims = chunks[0].embedding.length;
    const rows = chunks.map((c) => ({
      ...c,
      embedding: Array.from(c.embedding),
      tags: tag ? [tag] : c.tags,
    }));
    await this.table.add(rows);
  }

  async queryByVector(vector: Float32Array, k: number, filter?: QueryFilter): Promise<QueryResult[]> {
    await this.init();
    let q = await this.table.search(Array.from(vector)).limit(k);
    if (filter?.pathPrefix) q = q.where(`path like '${filter.pathPrefix}%'`);
    // LanceDB JS API may differ; fallback to getting all and filtering client-side if needed.
    const results = await q.execute();
    return results.map((r: any) => ({
      record: {
        id: r.id, path: r.path, startLine: r.startLine, endLine: r.endLine, content: r.content, lang: r.lang, tags: r.tags, hash: r.hash, ingestedAt: r.ingestedAt,
      },
      score: r._distance != null ? 1 - r._distance : r._score ?? 0,
    }));
  }

  async deleteByTag(tag: string): Promise<void> {
    await this.init();
    // Not all APIs support delete; as a simple approach, rebuild table would be needed.
    // For MVP, no-op or future implementation.
  }

  async stats(): Promise<{ count: number; dims: number | null }> {
    await this.init();
    const cnt = await this.table.count();
    return { count: cnt, dims: this.dims };
  }
}


