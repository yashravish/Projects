import { type NextRequest, NextResponse } from "next/server"
import { createServerClient } from "@/lib/supabase/server"
import { REPLICATE_API_TOKEN } from "@/lib/env"
import { checkStatusSchema } from "@/lib/validation"
import { rateLimiter, RATE_LIMITS } from "@/lib/rate-limit"

export async function GET(request: NextRequest) {
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

    // Rate limit check per user (more lenient for status checks)
    const rateLimitResult = rateLimiter.check(
      `status:${user.id}`,
      RATE_LIMITS.API.limit,
      RATE_LIMITS.API.window
    )
    if (!rateLimitResult.success) {
      return NextResponse.json(
        { error: "Too many requests. Please try again later." },
        { status: 429 }
      )
    }

    // Get generationId from query params
    const { searchParams } = new URL(request.url)
    const generationId = searchParams.get("generationId")

    // Validate input
    const validation = checkStatusSchema.safeParse({ generationId })
    if (!validation.success) {
      return NextResponse.json(
        { error: validation.error.errors[0].message },
        { status: 400 }
      )
    }

    // Fetch the generation record
    const { data: generation, error: generationError } = await supabase
      .from("song_generations")
      .select("*, songs!inner(user_id)")
      .eq("id", generationId)
      .single()

    if (generationError || !generation) {
      return NextResponse.json({ error: "Generation not found" }, { status: 404 })
    }

    // Verify ownership through the songs table
    if (generation.songs.user_id !== user.id) {
      return NextResponse.json({ error: "Access denied" }, { status: 403 })
    }

    // If already completed, return cached result
    if (generation.status === "succeeded" && generation.audio_url) {
      return NextResponse.json({
        status: generation.status,
        audio_url: generation.audio_url,
        generationId: generation.id,
      })
    }

    // Get prediction ID from raw_response
    const predictionId = generation.raw_response?.prediction_id

    if (!predictionId) {
      return NextResponse.json({ error: "Invalid generation record: missing prediction ID" }, { status: 500 })
    }

    // Call Replicate API to get prediction status
    const replicateResponse = await fetch(`https://api.replicate.com/v1/predictions/${predictionId}`, {
      headers: {
        Authorization: `Token ${REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
    })

    if (!replicateResponse.ok) {
      const errorText = await replicateResponse.text()
      if (process.env.NODE_ENV === "development") {
        console.error("Replicate status check error:", errorText)
      }
      return NextResponse.json({ error: "Failed to check generation status" }, { status: 500 })
    }

    const prediction = await replicateResponse.json()

    // Update generation record based on prediction status
    const updateData: any = {
      status: prediction.status,
      updated_at: new Date().toISOString(),
      raw_response: {
        ...generation.raw_response,
        latest_prediction: prediction,
      },
    }

    // If succeeded, extract audio URL
    if (prediction.status === "succeeded" && prediction.output) {
      // Replicate musicgen returns audio URL in output
      const audioUrl = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output

      updateData.audio_url = audioUrl
    }

    // If failed, store error
    if (prediction.status === "failed") {
      updateData.raw_response.error = prediction.error
    }

    // Update the generation record
    const { error: updateError } = await supabase.from("song_generations").update(updateData).eq("id", generationId)

    if (updateError && process.env.NODE_ENV === "development") {
      console.error("Failed to update generation:", updateError)
    }

    return NextResponse.json({
      status: prediction.status,
      audio_url: updateData.audio_url || null,
      generationId: generation.id,
      error: prediction.error || null,
    })
  } catch (error) {
    if (process.env.NODE_ENV === "development") {
      console.error("Status check error:", error)
    }
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
