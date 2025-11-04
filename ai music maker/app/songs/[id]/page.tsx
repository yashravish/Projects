import { notFound, redirect } from "next/navigation"
import { createClient } from "@/lib/supabase/server"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { AudioGenerationSection } from "@/components/audio-generation-section"

interface SongPageProps {
  params: Promise<{
    id: string
  }>
}

export default async function SongPage({ params }: SongPageProps) {
  const { id } = await params
  const supabase = await createClient()

  // Check authentication
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth")
  }

  // Fetch the song with its generations
  const { data: song, error } = await supabase
    .from("songs")
    .select(
      `
      *,
      song_generations (
        id,
        provider,
        status,
        audio_url,
        created_at,
        updated_at
      )
    `,
    )
    .eq("id", id)
    .eq("user_id", user.id)
    .order("created_at", { referencedTable: "song_generations", ascending: false })
    .single()

  if (error || !song) {
    notFound()
  }

  // Get the latest generation
  const latestGeneration = song.song_generations?.[0]

  return (
    <main className="min-h-screen bg-[#0f0f10] text-slate-100">
      <div className="mx-auto max-w-6xl px-5 py-6 space-y-6">
        <div className="flex items-center justify-between">
          <Button asChild variant="neutral">
            <Link href="/">← Back to Dashboard</Link>
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Song Metadata */}
          <div className="space-y-6">
            <Card className="rounded-lg border border-slate-700 bg-slate-900/50 backdrop-blur-sm">
              <CardHeader className="border-b border-slate-800">
                <CardTitle className="text-2xl text-slate-100">
                  {song.title}
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Created {new Date(song.created_at).toLocaleDateString()}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6 pt-6">
                {song.prompt && (
                  <div>
                    <h3 className="text-sm font-medium text-slate-300 mb-2">Description</h3>
                    <p className="text-sm text-slate-400 leading-relaxed">{song.prompt}</p>
                  </div>
                )}

                <Separator className="bg-slate-800" />

                <div className="grid grid-cols-2 gap-4">
                  {song.genre && (
                    <div>
                      <h4 className="text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                        Genre
                      </h4>
                      <p className="text-sm text-slate-200">{song.genre}</p>
                    </div>
                  )}
                  {song.mood && (
                    <div>
                      <h4 className="text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                        Mood
                      </h4>
                      <p className="text-sm text-slate-200">{song.mood}</p>
                    </div>
                  )}
                  {song.key && (
                    <div>
                      <h4 className="text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                        Key
                      </h4>
                      <p className="text-sm text-slate-200">{song.key}</p>
                    </div>
                  )}
                  {song.tempo && (
                    <div>
                      <h4 className="text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                        Tempo
                      </h4>
                      <p className="text-sm text-slate-200">{song.tempo} BPM</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column: Generation Panel and Audio Player */}
          <div className="space-y-6">
            <AudioGenerationSection
              songId={song.id}
              initialAudioUrl={latestGeneration?.audio_url}
              initialStatus={latestGeneration?.status}
            />
          </div>
        </div>
      </div>
    </main>
  )
}
