import { type NextRequest, NextResponse } from "next/server"
import { createServerClient } from "@/lib/supabase/server"
import { REPLICATE_API_TOKEN, REPLICATE_MUSIC_MODEL, OPENAI_API_KEY } from "@/lib/env"
import { generateAudioSchema } from "@/lib/validation"
import { rateLimiter, RATE_LIMITS } from "@/lib/rate-limit"

export async function POST(request: NextRequest) {
  try {
    const supabase = await createServerClient()

    // Check authentication
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()
    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Rate limit check per user
    const rateLimitResult = rateLimiter.check(
      `audio:${user.id}`,
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
    const validation = generateAudioSchema.safeParse(body)

    if (!validation.success) {
      return NextResponse.json(
        { error: validation.error.errors[0].message },
        { status: 400 }
      )
    }

    const { songId, duration: requestedDuration, quality: requestedQuality } = validation.data

    // Fetch the song from Supabase and verify ownership
    const { data: song, error: songError } = await supabase
      .from("songs")
      .select("*")
      .eq("id", songId)
      .eq("user_id", user.id)
      .single()

    if (songError || !song) {
      return NextResponse.json({ error: "Song not found or access denied" }, { status: 404 })
    }

    // Build an enriched prompt with OpenAI (no lyrics required)
    let musicPrompt = `${song.genre || "pop"} music, ${song.mood || "upbeat"} mood. ${song.prompt || ""}`

    try {
      if (OPENAI_API_KEY) {
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
                content: `Title: ${song.title || "Untitled"}\nGenre: ${song.genre || ""}\nMood: ${song.mood || ""}\nUser notes: ${song.prompt || ""}`,
              },
            ],
            temperature: 0.7,
            max_tokens: 120,
          }),
        })

        if (openaiRes.ok) {
          const data = await openaiRes.json()
          const text = data?.choices?.[0]?.message?.content?.trim()
          if (text) musicPrompt = text
        } else if (process.env.NODE_ENV === "development") {
          const errTxt = await openaiRes.text().catch(() => "")
          console.error("OpenAI prompt error:", errTxt)
        }
      }
    } catch (e) {
      if (process.env.NODE_ENV === "development") {
        console.error("OpenAI prompt generation failed:", e)
      }
    }

    // Resolve model version (support both "owner/model:version" and raw version id)
    const versionId = REPLICATE_MUSIC_MODEL.includes(":")
      ? REPLICATE_MUSIC_MODEL.split(":")[1]
      : REPLICATE_MUSIC_MODEL

    // Map requested quality to model_version param used by the model (best-effort)
    const allowedQualities = new Set(["stereo-melody-large", "stereo-large", "melody-large", "large"]) as Set<string>
    const modelVersion = requestedQuality && allowedQualities.has(requestedQuality) ? requestedQuality : "stereo-large"

    // Clamp duration between 5 and 30 seconds, default 15
    const duration = requestedDuration !== undefined && Number.isFinite(requestedDuration) ? Math.max(5, Math.min(30, requestedDuration)) : 15

    // Call Replicate API to create prediction
    const replicateResponse = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        Authorization: `Token ${REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: versionId,
        input: {
          prompt: musicPrompt,
          model_version: modelVersion,
          output_format: "mp3",
          normalization_strategy: "peak",
          duration,
        },
      }),
    })

    if (!replicateResponse.ok) {
      const errorText = await replicateResponse.text()
      if (process.env.NODE_ENV === "development") {
        console.error("Replicate API error:", errorText)
      }
      return NextResponse.json({ error: "Failed to start audio generation" }, { status: 500 })
    }

    const prediction = await replicateResponse.json()

    // Create song_generations row with status queued
    const { data: generation, error: generationError } = await supabase
      .from("song_generations")
      .insert({
        song_id: songId,
        provider: "replicate",
        status: "queued",
        raw_response: {
          prediction_id: prediction.id,
          model: REPLICATE_MUSIC_MODEL,
          prompt: musicPrompt,
        },
      })
      .select()
      .single()

    if (generationError) {
      if (process.env.NODE_ENV === "development") {
        console.error("Failed to create generation record:", generationError)
      }
      return NextResponse.json({ error: "Failed to save generation record" }, { status: 500 })
    }

    return NextResponse.json({
      generationId: generation.id,
      status: "queued",
      predictionId: prediction.id,
    })
  } catch (error) {
    if (process.env.NODE_ENV === "development") {
      console.error("Audio generation error:", error)
    }
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
