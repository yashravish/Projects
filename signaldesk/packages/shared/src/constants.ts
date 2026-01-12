export const CONFIG = {
  // Chunking
  CHUNK_SIZE_CHARS: 1200,
  CHUNK_OVERLAP_CHARS: 200,

  // RAG
  RETRIEVAL_TOP_K: 6,
  MAX_CONTEXT_TOKENS: 4000,

  // Models
  EMBEDDING_MODEL: 'text-embedding-3-small',
  EMBEDDING_DIMENSIONS: 1536,
  CHAT_MODEL: 'gpt-4o-mini',
  CHAT_MAX_TOKENS: 1024,
  CHAT_TEMPERATURE: 0, // Deterministic

  // Upload
  MAX_FILE_SIZE_BYTES: 10 * 1024 * 1024, // 10MB
  ALLOWED_MIME_TYPES: ['application/pdf', 'text/plain', 'text/markdown'] as const,
  ALLOWED_EXTENSIONS: ['.pdf', '.txt', '.md'] as const,

  // Auth
  JWT_EXPIRY: '7d',
  BCRYPT_ROUNDS: 12,

  // Rate Limiting
  AUTH_RATE_LIMIT_WINDOW_MS: 15 * 60 * 1000, // 15 minutes
  AUTH_RATE_LIMIT_MAX: 10, // 10 attempts per window

  // Token Estimation (for cost analytics)
  CHARS_PER_TOKEN_ESTIMATE: 4,
  EMBEDDING_COST_PER_1M_TOKENS: 0.02,
  CHAT_INPUT_COST_PER_1M_TOKENS: 0.15,
  CHAT_OUTPUT_COST_PER_1M_TOKENS: 0.6,
} as const;
