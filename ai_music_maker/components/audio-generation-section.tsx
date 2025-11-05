"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface AudioGenerationSectionProps {
  songId: string
  initialAudioUrl?: string | null
  initialStatus?: string | null
}

export function AudioGenerationSection({ songId, initialAudioUrl, initialStatus }: AudioGenerationSectionProps) {
  const [isGenerating, setIsGenerating] = useState(false)
  const [status, setStatus] = useState<string>(initialStatus || "")
  const [audioUrl, setAudioUrl] = useState<string | null>(initialAudioUrl || null)
  const [error, setError] = useState<string | null>(null)
  const [generationId, setGenerationId] = useState<string | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const [duration, setDuration] = useState<number>(15)
  const [quality, setQuality] = useState<string>("stereo-large")

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    }
  }, [])

  const pollStatus = async (genId: string) => {
    try {
      const response = await fetch(`/api/generate/status?generationId=${genId}`)

      if (!response.ok) {
        const data = await response.json().catch(() => ({ error: "Failed to check status" }))
        throw new Error(data.error || "Failed to check status")
      }

      const data = await response.json()

      setStatus(data.status)

      if (data.audio_url) {
        setAudioUrl(data.audio_url)
      }

      // Stop polling if generation is complete or failed
      if (data.status === "succeeded" || data.status === "failed") {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        setIsGenerating(false)

        if (data.status === "failed") {
          setError("Audio generation failed. Please try again.")
        }
      }
    } catch (err) {
      if (process.env.NODE_ENV === "development") {
        console.error("Error polling status:", err)
      }
      setError(err instanceof Error ? err.message : "Failed to check generation status")
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
      setIsGenerating(false)
    }
  }

  const handleGenerateAudio = async () => {
    setIsGenerating(true)
    setError(null)
    setStatus("starting")

    try {
      const response = await fetch("/api/generate/audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ songId, duration, quality }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => ({ error: "Failed to start audio generation" }))
        throw new Error(data.error || "Failed to start audio generation")
      }

      const data = await response.json()

      setGenerationId(data.generationId)
      setStatus("queued")

      // Start polling every 3 seconds
      pollingIntervalRef.current = setInterval(() => {
        pollStatus(data.generationId)
      }, 3000)

      // Do an immediate poll
      pollStatus(data.generationId)
    } catch (err) {
      if (process.env.NODE_ENV === "development") {
        console.error("Error generating audio:", err)
      }
      setError(err instanceof Error ? err.message : "Failed to generate audio")
      setIsGenerating(false)
    }
  }

  const getStatusDisplay = (status: string) => {
    switch (status) {
      case "queued":
        return "Queued"
      case "starting":
      case "processing":
        return "Running"
      case "succeeded":
        return "Complete"
      case "failed":
        return "Failed"
      default:
        return status
    }
  }

  return (
    <div className="space-y-6">
      {audioUrl && (
        <Card className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md">
          <CardHeader>
            <CardTitle>Audio Player</CardTitle>
            <CardDescription>Your generated song</CardDescription>
          </CardHeader>
          <CardContent>
            <audio controls className="w-full">
              <source src={audioUrl} type="audio/mpeg" />
              Your browser does not support the audio element.
            </audio>
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md">
        <CardHeader className="px-5">
          <div className="flex items-center justify-between">
            <CardTitle>Audio Generation</CardTitle>
            {status && (
              <Badge variant={status === "succeeded" ? "default" : status === "failed" ? "destructive" : "secondary"}>
                {getStatusDisplay(status)}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4 px-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-sm text-slate-400">Duration (sec)</label>
              <input
                type="number"
                min={5}
                max={30}
                step={5}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="mt-1 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-white/20"
                disabled={isGenerating}
              />
            </div>
            <div>
              <label className="text-sm text-slate-400">Quality</label>
              <select
                value={quality}
                onChange={(e) => setQuality(e.target.value)}
                className="mt-1 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-white/20"
                disabled={isGenerating}
              >
                <option value="large">Faster (large)</option>
                <option value="stereo-large">Balanced (stereo-large)</option>
                <option value="melody-large">Melody focused (melody-large)</option>
                <option value="stereo-melody-large">Premium (stereo-melody-large)</option>
              </select>
            </div>
          </div>
          <Button onClick={handleGenerateAudio} disabled={isGenerating} className="w-full">
            {isGenerating ? "Generating..." : "Generate audio"}
          </Button>

          {error && <p className="text-sm text-destructive">{error}</p>}

          {isGenerating && (
            <div className="text-sm text-muted-foreground">
              <p>Status: {getStatusDisplay(status)}</p>
              <p className="text-xs mt-1">This may take a few minutes...</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
