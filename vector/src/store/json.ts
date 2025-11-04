import fs from 'node:fs/promises';
import path from 'node:path';
import { expandHome } from '../config.js';
import type { VectorStore, ChunkRecord, QueryFilter, QueryResult } from './types.js';

type Persisted = Omit<ChunkRecord, 'embedding'> & { embedding: number[] };

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

export class JsonVectorStore implements VectorStore {
  private dbFile: string;
  constructor(dbPath: string) {
    const abs = expandHome(dbPath);
    this.dbFile = path.join(abs, 'chunks.json');
  }

  private async readAll(): Promise<Persisted[]> {
    try {
      const raw = await fs.readFile(this.dbFile, 'utf8');
      return JSON.parse(raw);
    } catch {
      return [];
    }
  }

  private async writeAll(rows: Persisted[]): Promise<void> {
    await fs.mkdir(path.dirname(this.dbFile), { recursive: true });
    await fs.writeFile(this.dbFile, JSON.stringify(rows), 'utf8');
  }

  async upsertChunks(chunks: ChunkRecord[], tag?: string): Promise<void> {
    const rows = await this.readAll();
    const byId = new Map(rows.map((r) => [r.id, r] as const));
    for (const ch of chunks) {
      const persisted: Persisted = { ...ch, embedding: Array.from(ch.embedding), tags: tag ? [tag] : ch.tags } as any;
      byId.set(ch.id, persisted);
    }
    await this.writeAll(Array.from(byId.values()));
  }

  async queryByVector(vector: Float32Array, k: number, filter?: QueryFilter): Promise<QueryResult[]> {
    const rows = await this.readAll();
    const candidates = rows.filter((r) => {
      if (filter?.tag && !(r.tags || []).includes(filter.tag)) return false;
      if (filter?.pathPrefix && !r.path.startsWith(filter.pathPrefix)) return false;
      return true;
    });
    const vec = vector;
    const scored = candidates.map((r) => {
      const emb = new Float32Array(r.embedding);
      return { r, s: cosine(vec, emb) };
    });
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, k).map(({ r, s }) => ({
      record: { id: r.id, path: r.path, startLine: r.startLine, endLine: r.endLine, content: r.content, lang: r.lang, tags: r.tags, hash: r.hash, ingestedAt: r.ingestedAt },
      score: s,
    }));
  }

  async deleteByTag(tag: string): Promise<void> {
    const rows = await this.readAll();
    const next = rows.filter((r) => !(r.tags || []).includes(tag));
    await this.writeAll(next);
  }

  async stats(): Promise<{ count: number; dims: number | null }> {
    const rows = await this.readAll();
    const dims = rows.length > 0 ? rows[0].embedding.length : null;
    return { count: rows.length, dims };
  }
}


