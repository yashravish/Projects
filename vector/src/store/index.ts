import { LanceDBStore, probeLanceDB } from './lancedb.js';
import { JsonVectorStore } from './json.js';
import type { VectorStore } from './types.js';

export async function openVectorStore(dbPath: string, prefer?: 'lancedb' | 'json'): Promise<VectorStore> {
  if (prefer === 'json') return new JsonVectorStore(dbPath);
  if (prefer === 'lancedb') {
    const ok = await probeLanceDB();
    if (!ok) throw new Error('LanceDB requested but not available');
    return new LanceDBStore(dbPath);
  }
  // Auto probe
  if (await probeLanceDB()) return new LanceDBStore(dbPath);
  return new JsonVectorStore(dbPath);
}


