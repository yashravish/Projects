/**
 * Simple in-memory rate limiter for API routes
 * For production, consider using Redis-based solution like @upstash/ratelimit
 */

interface RateLimitEntry {
  count: number
  resetAt: number
}

class RateLimiter {
  private requests: Map<string, RateLimitEntry> = new Map()
  private cleanupInterval: NodeJS.Timeout | null = null

  constructor() {
    // Clean up expired entries every 5 minutes
    if (typeof window === "undefined") {
      this.cleanupInterval = setInterval(() => {
        const now = Date.now()
        for (const [key, entry] of this.requests.entries()) {
          if (now > entry.resetAt) {
            this.requests.delete(key)
          }
        }
      }, 5 * 60 * 1000)
    }
  }

  /**
   * Check if request should be rate limited
   * @param identifier - Unique identifier (user ID, IP, etc.)
   * @param limit - Maximum requests allowed
   * @param windowMs - Time window in milliseconds
   * @returns Object with success status and remaining/reset info
   */
  check(
    identifier: string,
    limit: number,
    windowMs: number
  ): { success: boolean; remaining: number; reset: number } {
    const now = Date.now()
    const entry = this.requests.get(identifier)

    // No entry or expired - create new
    if (!entry || now > entry.resetAt) {
      this.requests.set(identifier, {
        count: 1,
        resetAt: now + windowMs,
      })
      return {
        success: true,
        remaining: limit - 1,
        reset: now + windowMs,
      }
    }

    // Entry exists and not expired
    if (entry.count >= limit) {
      return {
        success: false,
        remaining: 0,
        reset: entry.resetAt,
      }
    }

    // Increment count
    entry.count++
    this.requests.set(identifier, entry)

    return {
      success: true,
      remaining: limit - entry.count,
      reset: entry.resetAt,
    }
  }

  cleanup() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval)
    }
    this.requests.clear()
  }
}

// Singleton instance
export const rateLimiter = new RateLimiter()

/**
 * Rate limit configurations for different endpoints
 */
export const RATE_LIMITS = {
  // AI generation endpoints - more restrictive
  AI_GENERATION: {
    limit: 10, // 10 requests
    window: 60 * 60 * 1000, // per hour
  },
  // Auth endpoints - moderate
  AUTH: {
    limit: 5, // 5 requests
    window: 15 * 60 * 1000, // per 15 minutes
  },
  // General API - lenient
  API: {
    limit: 100, // 100 requests
    window: 15 * 60 * 1000, // per 15 minutes
  },
} as const
