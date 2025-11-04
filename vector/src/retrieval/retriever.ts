import type { VectorStore, QueryResult } from '../store/types.js';

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

export function mmrDiversify(
  candidates: { vec: Float32Array; item: QueryResult }[],
  k: number,
  lambda: number
): QueryResult[] {
  if (candidates.length <= k) return candidates.map((c) => c.item);
  const selected: { vec: Float32Array; item: QueryResult }[] = [];
  const remaining = candidates.slice();
  // Seed with top-scored
  remaining.sort((a, b) => b.item.score - a.item.score);
  selected.push(remaining.shift()!);
  while (selected.length < k && remaining.length > 0) {
    let bestIdx = 0; let bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];
      const relevance = candidate.item.score;
      let diversity = 0;
      for (const s of selected) diversity = Math.max(diversity, cosine(candidate.vec, s.vec));
      const mmr = lambda * relevance - (1 - lambda) * diversity;
      if (mmr > bestScore) { bestScore = mmr; bestIdx = i; }
    }
    selected.push(remaining.splice(bestIdx, 1)[0]);
  }
  return selected.map((s) => s.item);
}

export type RetrieveOptions = {
  k: number;
  lambda: number;
  minScore: number;
  filterPathPrefix?: string;
};

export async function retrieve(
  store: VectorStore,
  queryEmbedding: Float32Array,
  opts: RetrieveOptions
): Promise<QueryResult[]> {
  const prelim = await store.queryByVector(queryEmbedding, Math.max(opts.k * 2, 50), {
    pathPrefix: opts.filterPathPrefix,
  });
  const filtered = prelim.filter((r) => r.score >= opts.minScore);
  const withVecs = filtered.map((r) => ({ vec: queryEmbedding, item: r }));
  const diversified = mmrDiversify(withVecs, opts.k, opts.lambda);
  return diversified;
}


