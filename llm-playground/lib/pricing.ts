import { LIMITS } from "./constants";

const PRICING: Record<string, { inPerM: number; outPerM: number }> = {
  "gpt-4o-mini": { inPerM: 0.00015, outPerM: 0.00060 }
};

export function estimateCostUSD(model: string, inTok: number, outTok: number): number {
  const p = PRICING[model];

  if (!p) {
    console.warn(`[pricing] No pricing entry for model "${model}". Using fallback rates.`);
  }

  const inPerM = p?.inPerM ?? 0.00015;
  const outPerM = p?.outPerM ?? 0.00060;

  return (inTok * inPerM + outTok * outPerM) / 1_000_000;
}

/**
 * Token estimation using character-based heuristic.
 * Approximates ~4 characters per token (OpenAI's rough estimate).
 *
 * Note: For production accuracy, consider using tiktoken in a server-only context
 * or an API-based token counter.
 */
export function roughTokenEstimate(text: string): number {
  return Math.max(1, Math.round(text.trim().length / LIMITS.TOKEN_ESTIMATE_RATIO));
}
