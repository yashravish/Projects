import { z } from "zod"

/**
 * Validation schemas for API endpoints
 */

// Auth schemas
export const signUpSchema = z.object({
  email: z.string().email("Invalid email address").max(255, "Email too long"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .max(128, "Password too long"),
})

// Song creation schema
export const createSongSchema = z.object({
  title: z.string().min(1, "Title is required").max(200, "Title too long"),
  prompt: z.string().min(1, "Prompt is required").max(1000, "Prompt too long"),
  genre: z.string().max(100, "Genre too long").optional(),
  mood: z.string().max(100, "Mood too long").optional(),
  key: z.string().max(50, "Key too long").optional(),
  tempo: z.number().int().min(40).max(200).optional(),
})

// Lyrics generation schema
export const generateLyricsSchema = z.object({
  songId: z.string().uuid("Invalid song ID"),
  prompt: z.string().max(1000, "Prompt too long").optional(),
  genre: z.string().max(100, "Genre too long").optional(),
  mood: z.string().max(100, "Mood too long").optional(),
})

// Audio generation schema
export const generateAudioSchema = z.object({
  songId: z.string().uuid("Invalid song ID"),
  duration: z.number().int().min(5, "Duration must be at least 5 seconds").max(30, "Duration must be at most 30 seconds").optional(),
  quality: z.enum(["large", "stereo-large", "melody-large", "stereo-melody-large"]).optional(),
})

// Status check schema
export const checkStatusSchema = z.object({
  generationId: z.string().uuid("Invalid generation ID"),
})

/**
 * Type exports for TypeScript
 */
export type SignUpInput = z.infer<typeof signUpSchema>
export type CreateSongInput = z.infer<typeof createSongSchema>
export type GenerateLyricsInput = z.infer<typeof generateLyricsSchema>
export type GenerateAudioInput = z.infer<typeof generateAudioSchema>
export type CheckStatusInput = z.infer<typeof checkStatusSchema>
