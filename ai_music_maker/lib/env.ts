/**
 * Environment variable validation and type-safe access
 * Throws descriptive errors if required variables are missing
 */

function getEnvVar(key: string, required = true): string {
  const value = process.env[key]

  if (!value && required) {
    throw new Error(
      `Missing required environment variable: ${key}\n` +
        `Please add it to your .env.local file or deployment environment.\n` +
        `See .env.example for reference.`,
    )
  }

  return value || ""
}

// Supabase
export const SUPABASE_URL = getEnvVar("NEXT_PUBLIC_SUPABASE_URL")
export const SUPABASE_ANON_KEY = getEnvVar("NEXT_PUBLIC_SUPABASE_ANON_KEY")
export const SUPABASE_SERVICE_ROLE_KEY = getEnvVar("SUPABASE_SERVICE_ROLE_KEY", false)

// OpenAI
export const OPENAI_API_KEY = getEnvVar("OPENAI_API_KEY")

// Replicate
export const REPLICATE_API_TOKEN = getEnvVar("REPLICATE_API_TOKEN")
export const REPLICATE_MUSIC_MODEL = getEnvVar("REPLICATE_MUSIC_MODEL")

// Site
export const SITE_URL = getEnvVar("NEXT_PUBLIC_SITE_URL", false) || "http://localhost:3000"

// Validate all required env vars on import
if (typeof window === "undefined") {
  // Only validate on server-side
  const requiredVars = {
    NEXT_PUBLIC_SUPABASE_URL: SUPABASE_URL,
    NEXT_PUBLIC_SUPABASE_ANON_KEY: SUPABASE_ANON_KEY,
    OPENAI_API_KEY: OPENAI_API_KEY,
    REPLICATE_API_TOKEN: REPLICATE_API_TOKEN,
    REPLICATE_MUSIC_MODEL: REPLICATE_MUSIC_MODEL,
  }

  const missing = Object.entries(requiredVars)
    .filter(([_, value]) => !value)
    .map(([key]) => key)

  if (missing.length > 0) {
    console.error(
      `❌ Missing required environment variables:\n${missing.map((k) => `  - ${k}`).join("\n")}\n` +
        `Please check your .env.local file or deployment environment.`,
    )
  }
}
