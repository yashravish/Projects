// Application-wide constants

export const LIMITS = {
  PROMPT_MAX_LENGTH: 16_000,
  MAX_TOKENS_DEFAULT: 256,
  MAX_TOKENS_LIMIT: 8_192,
  RUNS_TABLE_LIMIT: 50,
  TOKEN_ESTIMATE_RATIO: 4,
} as const;

export const RATE_LIMITS = {
  REQUESTS_PER_MINUTE: 10,
  WINDOW_MS: 60_000,
} as const;

export const WEB_LIMITS = {
  WEB_MAX_RESULTS: 5,
  WEB_PER_PAGE_CHAR_CAP: 10_000,
  WEB_TOTAL_CONTEXT_CHAR_CAP: 20_000,
  REQUEST_TIMEOUT_MS: 12_000,
  CITED_PROMPT_MAX_LENGTH: 15_000, // Keep under 16k limit for /api/generate
} as const;

export const SUPPORT_LIMITS = {
  KB_TEXT_MAX_LENGTH: 60_000, // Max chars for knowledge base text
  CHUNK_TARGET_SIZE: 800, // Target size for each chunk
  CHUNK_OVERLAP: 120, // Overlap between chunks
  TOP_K_CHUNKS: 5, // Number of chunks to retrieve
  MIN_SCORE_THRESHOLD: 0.01, // Minimum relevance score
  SUPPORT_PROMPT_MAX_LENGTH: 14_000, // Leave room for instructions
} as const;
