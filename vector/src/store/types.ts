export type ChunkRecord = {
  id: string;
  path: string; // posix
  startLine: number;
  endLine: number;
  content: string;
  embedding: Float32Array;
  lang: string;
  tags?: string[];
  hash?: string;
  ingestedAt: number;
};

export type QueryFilter = {
  tag?: string;
  pathPrefix?: string;
};

export type QueryResult = {
  record: Omit<ChunkRecord, 'embedding'>;
  score: number; // cosine similarity
};

export interface VectorStore {
  upsertChunks(chunks: ChunkRecord[], tag?: string): Promise<void>;
  queryByVector(vector: Float32Array, k: number, filter?: QueryFilter): Promise<QueryResult[]>;
  deleteByTag(tag: string): Promise<void>;
  stats(): Promise<{ count: number; dims: number | null }>; 
}


