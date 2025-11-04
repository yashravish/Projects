import { LIMITS } from "./constants";

export function sanitizePrompt(prompt: string): string {
  // Remove control characters
  const cleaned = prompt.replace(/[\x00-\x1F\x7F]/g, "");

  // Limit length
  if (cleaned.length > LIMITS.PROMPT_MAX_LENGTH) {
    throw new Error(`Prompt too long (max ${LIMITS.PROMPT_MAX_LENGTH} characters)`);
  }

  // Check for suspicious patterns (optional - log only, don't block)
  const suspiciousPatterns = [
    /ignore previous instructions/i,
    /disregard all previous/i,
    /forget (all|everything) (you|above)/i,
  ];

  for (const pattern of suspiciousPatterns) {
    if (pattern.test(cleaned)) {
      console.warn("Suspicious prompt pattern detected:", cleaned.slice(0, 100));
      // Don't block - OpenAI has its own filters
      // Just log for monitoring purposes
    }
  }

  return cleaned.trim();
}
