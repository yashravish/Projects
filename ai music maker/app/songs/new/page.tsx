import { redirect } from "next/navigation"
import Link from "next/link"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

async function createSong(formData: FormData) {
  "use server"

  const supabase = await createClient()

  // Get current user
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth")
  }

  // Extract form data
  const title = formData.get("title") as string
  const prompt = formData.get("prompt") as string
  const genre = formData.get("genre") as string
  const mood = formData.get("mood") as string
  const key = formData.get("key") as string
  const tempo = formData.get("tempo") ? Number.parseInt(formData.get("tempo") as string) : null

  // Insert song into database
  const { data: song, error } = await supabase
    .from("songs")
    .insert({
      user_id: user.id,
      title,
      prompt,
      genre,
      mood,
      key,
      tempo,
    })
    .select()
    .single()

  if (error) {
    if (process.env.NODE_ENV === "development") {
      console.error("Error creating song:", error)
    }
    throw new Error(error.message || "Failed to create song")
  }

  // Redirect to the song detail page
  redirect(`/songs/${song.id}`)
}

export default async function NewSongPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth")
  }

  return (
    <div className="container mx-auto py-8 px-4 min-h-screen">
      <Card className="max-w-2xl mx-auto border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <CardHeader className="border-b border-slate-800">
          <CardTitle className="text-2xl text-slate-100">
            Create New Song
          </CardTitle>
          <CardDescription className="text-slate-400">Fill in the details to generate your AI song</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <form action={createSong} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="title" className="text-slate-300">
                Title
              </Label>
              <Input
                id="title"
                name="title"
                placeholder="Enter song title"
                required
                className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="prompt" className="text-slate-300">
                Description / Prompt
              </Label>
              <Textarea
                id="prompt"
                name="prompt"
                placeholder="Describe the song you want to create..."
                rows={4}
                required
                className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors resize-none"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="genre" className="text-slate-300">
                  Genre
                </Label>
                <Input
                  id="genre"
                  name="genre"
                  placeholder="e.g., Pop, Rock, Jazz"
                  className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="mood" className="text-slate-300">
                  Mood
                </Label>
                <Input
                  id="mood"
                  name="mood"
                  placeholder="e.g., Happy, Melancholic"
                  className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="key" className="text-slate-300">
                  Key
                </Label>
                <Input
                  id="key"
                  name="key"
                  placeholder="e.g., C Major, A Minor"
                  className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="tempo" className="text-slate-300">
                  Tempo (BPM)
                </Label>
                <Input
                  id="tempo"
                  name="tempo"
                  type="number"
                  placeholder="e.g., 120"
                  min="40"
                  max="200"
                  className="bg-slate-800 border-slate-700 focus:border-slate-600 transition-colors"
                />
              </div>
            </div>

            <div className="flex justify-end gap-4 pt-4">
              <Button asChild variant="neutral" className="transition-colors">
                <Link href="/">Cancel</Link>
              </Button>
              <Button
                type="submit"
                className="bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors"
              >
                Create Song
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
