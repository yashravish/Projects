"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

interface LyricsSectionProps {
  songId: string
  initialLyrics: string | null
  prompt: string | null
  genre: string | null
  mood: string | null
}

export function LyricsSection({ songId, initialLyrics, prompt, genre, mood }: LyricsSectionProps) {
  const [lyrics, setLyrics] = useState(initialLyrics)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const handleGenerateLyrics = async () => {
    setIsGenerating(true)
    setError(null)

    try {
      const response = await fetch("/api/generate/lyrics", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          songId,
          prompt: prompt || "",
          genre: genre || "",
          mood: mood || "",
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Failed to generate lyrics" }))
        throw new Error(errorData.error || "Failed to generate lyrics")
      }

      const data = await response.json()
      setLyrics(data.fullLyrics)
      router.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <Card className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-slate-100">Lyrics</CardTitle>
          <Button onClick={handleGenerateLyrics} disabled={isGenerating} variant="outline" size="sm">
            {isGenerating ? "Generating..." : lyrics ? "Regenerate Lyrics" : "Generate Lyrics"}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {error && <div className="mb-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">{error}</div>}
        {lyrics ? (
          <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans text-slate-200">{lyrics}</pre>
        ) : (
          <p className="text-sm text-slate-400">No lyrics yet. Click "Generate Lyrics" to create them.</p>
        )}
      </CardContent>
    </Card>
  )
}
