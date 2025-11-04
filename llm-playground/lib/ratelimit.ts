// Simple in-memory rate limiting
// For production, consider using Upstash Redis or similar

import { RATE_LIMITS } from "./constants";

const requests = new Map<string, number[]>();

export function checkRateLimit(
  identifier: string,
  maxRequests = RATE_LIMITS.REQUESTS_PER_MINUTE,
  windowMs = RATE_LIMITS.WINDOW_MS
): { success: boolean; remaining: number; reset: number } {
  const now = Date.now();
  const userRequests = requests.get(identifier) || [];

  // Remove old requests outside the window
  const recentRequests = userRequests.filter((time) => now - time < windowMs);

  if (recentRequests.length >= maxRequests) {
    const oldestRequest = Math.min(...recentRequests);
    return {
      success: false,
      remaining: 0,
      reset: oldestRequest + windowMs,
    };
  }

  recentRequests.push(now);
  requests.set(identifier, recentRequests);

  return {
    success: true,
    remaining: maxRequests - recentRequests.length,
    reset: now + windowMs,
  };
}

// Cleanup old entries periodically to prevent memory leaks
if (typeof setInterval !== "undefined") {
  setInterval(() => {
    const now = Date.now();
    for (const [key, times] of requests.entries()) {
      const recent = times.filter((t) => now - t < RATE_LIMITS.WINDOW_MS * 2);
      if (recent.length === 0) {
        requests.delete(key);
      } else {
        requests.set(key, recent);
      }
    }
  }, 5 * 60 * 1000); // Clean up every 5 minutes
}
