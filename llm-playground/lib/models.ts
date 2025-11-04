// Whitelist allowed models to prevent typos and apply limits.
export const ALLOWED_MODELS = {
  "gpt-4o-mini": { ctx: 128_000, max_out: 8192 }
} as const;

export type AllowedModel = keyof typeof ALLOWED_MODELS;

// Safe default model with validation
export const DEFAULT_MODEL: AllowedModel = (() => {
  const envModel = process.env.NEXT_PUBLIC_DEFAULT_MODEL;
  if (envModel && envModel in ALLOWED_MODELS) {
    return envModel as AllowedModel;
  }
  return "gpt-4o-mini";
})();

// Export array for UI usage
export const MODELS_ARRAY = Object.keys(ALLOWED_MODELS) as AllowedModel[];
