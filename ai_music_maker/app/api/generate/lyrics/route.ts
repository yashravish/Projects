import { createServerClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"
import { OPENAI_API_KEY } from "@/lib/env"
import { generateLyricsSchema } from "@/lib/validation"
import { rateLimiter, RATE_LIMITS } from "@/lib/rate-limit"

interface LyricsSection {
  type: string
  content: string
}

interface LyricsResponse {
  fullLyrics: string
  sections: LyricsSection[]
  enhancedMusicPrompt?: string
}

// Simple structure placeholder - no actual lyrics content
function generateLyricsStructure(genre: string, mood: string): LyricsResponse {
  const sections: LyricsSection[] = [
    { type: "Intro", content: "Instrumental" },
    { type: "Verse 1", content: "..." },
    { type: "Pre-Chorus", content: "..." },
    { type: "Chorus", content: "..." },
    { type: "Verse 2", content: "..." },
    { type: "Chorus", content: "..." },
    { type: "Bridge", content: "..." },
    { type: "Chorus", content: "..." },
    { type: "Outro", content: "Instrumental" }
  ]

  const fullLyrics = `Song Structure for ${genre} - ${mood}

[Intro] - Instrumental
[Verse 1]
[Pre-Chorus]
[Chorus]
[Verse 2]
[Chorus]
[Bridge]
[Chorus]
[Outro] - Instrumental

Note: This is a structural guide. The AI will generate the music based on your prompt.`

  return { fullLyrics, sections }
}

// Use OpenAI to enhance the music generation prompt
async function enhanceMusicPrompt(prompt: string, genre: string, mood: string): Promise<string | null> {
  if (!OPENAI_API_KEY) {
    return null
  }

  try {
    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a music prompt engineer. Return a single concise line (max 40 words) describing genre, mood, instrumentation, production style, and vibe for an AI music model. Avoid quotes and special characters.",
          },
          {
            role: "user",
            content: `Genre: ${genre || "pop"}\nMood: ${mood || "upbeat"}\nUser idea: ${prompt || "instrumental track"}`,
          },
        ],
        temperature: 0.7,
        max_tokens: 120,
      }),
    })

    if (openaiRes.ok) {
      const data = await openaiRes.json()
      return data?.choices?.[0]?.message?.content?.trim() || null
    }
  } catch (e) {
    // Log error but don't fail the request
    if (process.env.NODE_ENV === "development") {
      console.error("OpenAI prompt enhancement failed:", e)
    }
  }

  return null
}

export async function POST(request: Request) {
  try {
    // Initialize Supabase client
    const supabase = await createServerClient()

    // Get current user
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()

    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Rate limit check per user
    const rateLimitResult = rateLimiter.check(
      `lyrics:${user.id}`,
      RATE_LIMITS.AI_GENERATION.limit,
      RATE_LIMITS.AI_GENERATION.window
    )
    if (!rateLimitResult.success) {
      return NextResponse.json(
        { error: "Too many generation requests. Please try again later." },
        {
          status: 429,
          headers: {
            "X-RateLimit-Limit": RATE_LIMITS.AI_GENERATION.limit.toString(),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": new Date(rateLimitResult.reset).toISOString(),
          }
        }
      )
    }

    // Parse and validate request body
    const body = await request.json()
    const validation = generateLyricsSchema.safeParse(body)

    if (!validation.success) {
      return NextResponse.json(
        { error: validation.error.errors[0].message },
        { status: 400 }
      )
    }

    const { songId, prompt, genre, mood } = validation.data

    // Fetch the song and verify ownership
    const { data: song, error: songError } = await supabase
      .from("songs")
      .select("*")
      .eq("id", songId)
      .eq("user_id", user.id)
      .single()

    if (songError || !song) {
      return NextResponse.json({ error: "Song not found or access denied" }, { status: 404 })
    }

    // Generate simple structure placeholder
    const { fullLyrics, sections } = generateLyricsStructure(
      genre || song.genre || "pop",
      mood || song.mood || "upbeat"
    )

    // Use OpenAI to enhance the music prompt (this is the real value)
    const enhancedPrompt = await enhanceMusicPrompt(
      prompt || song.prompt || "",
      genre || song.genre || "pop",
      mood || song.mood || "upbeat"
    )

    // Update the song with the structure and enhanced prompt
    const updateData: any = { lyrics: fullLyrics }
    if (enhancedPrompt) {
      updateData.enhanced_prompt = enhancedPrompt
    }

    const { error: updateError } = await supabase
      .from("songs")
      .update(updateData)
      .eq("id", songId)
      .eq("user_id", user.id)

    if (updateError) {
      if (process.env.NODE_ENV === "development") {
        console.error("Failed to update song:", updateError)
      }
      return NextResponse.json({ error: "Failed to save to database" }, { status: 500 })
    }

    // Return structured response with enhanced prompt
    const response: LyricsResponse = {
      fullLyrics,
      sections,
      enhancedMusicPrompt: enhancedPrompt || undefined,
    }

    return NextResponse.json(response, { status: 200 })
  } catch (error) {
    if (process.env.NODE_ENV === "development") {
      console.error("Error in lyrics generation:", error)
    }
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
