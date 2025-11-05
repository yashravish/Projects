import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import AuthInline from "@/components/auth-inline"

export default async function Home() {
  const supabase = await createClient()
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser()

  const isAuthenticated = !!user && !error

  async function signOut() {
    "use server"
    const sb = await createClient()
    await sb.auth.signOut()
    redirect("/")
  }

  const { data: songs } = isAuthenticated
    ? await supabase
    .from("songs")
    .select(
      `
      id,
      title,
      genre,
      created_at,
      song_generations (
        status,
        created_at
      )
    `,
    )
    .eq("user_id", user!.id)
    .order("created_at", { ascending: false })
    : { data: null }

  // Process songs to get latest generation status
  const songsWithStatus = songs?.map((song) => {
    const generations = song.song_generations as Array<{
      status: string
      created_at: string
    }>
    const latestGeneration = generations?.sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    )[0]

    return {
      id: song.id,
      title: song.title || "Untitled",
      genre: song.genre || "—",
      created_at: song.created_at,
      status: latestGeneration?.status || "—",
    }
  })

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <div className="max-w-6xl w-full space-y-6">
        {isAuthenticated ? (
          <div className="flex items-center justify-between animate-fade-in-down">
            <div className="space-y-1">
              <h1 className="text-4xl font-semibold tracking-tight text-slate-100">
                Your Songs
              </h1>
              <p className="text-slate-400">Manage and create AI-generated songs</p>
            </div>
            <div className="flex items-center gap-3">
              <Button asChild className="bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors">
                <Link href="/songs/new">New Song</Link>
              </Button>
              <form action={signOut}>
                <Button type="submit" variant="neutral" className="transition-colors">
                  Sign out
                </Button>
              </form>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center min-h-[80vh] animate-fade-in">
            <div className="w-full max-w-md">
              <AuthInline />
            </div>
          </div>
        )}

        {isAuthenticated && (
          <div className="rounded-lg border border-slate-700 bg-slate-900/50 backdrop-blur-sm animate-fade-in-up overflow-hidden">
            {songsWithStatus && songsWithStatus.length > 0 ? (
              <table className="w-full border-collapse text-left">
                <thead className="bg-slate-800/50 border-b border-slate-700">
                  <tr>
                    <th className="px-6 py-3 text-xs font-medium uppercase tracking-wider text-slate-400">Title</th>
                    <th className="px-6 py-3 text-xs font-medium uppercase tracking-wider text-slate-400">Genre</th>
                    <th className="px-6 py-3 text-xs font-medium uppercase tracking-wider text-slate-400">Created</th>
                    <th className="px-6 py-3 text-xs font-medium uppercase tracking-wider text-slate-400">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {songsWithStatus.map((song) => (
                    <tr
                      key={song.id}
                      className="group hover:bg-slate-800/50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <Link
                          href={`/songs/${song.id}`}
                          className="text-sm font-medium text-slate-100 hover:text-slate-300 transition-colors"
                        >
                          {song.title}
                        </Link>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-400">
                        {song.genre}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-400">{new Date(song.created_at).toLocaleDateString()}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium ${
                          song.status === 'succeeded'
                            ? 'bg-green-500/10 text-green-400 ring-1 ring-inset ring-green-500/20'
                            : song.status === 'failed'
                            ? 'bg-red-500/10 text-red-400 ring-1 ring-inset ring-red-500/20'
                            : song.status === 'processing' || song.status === 'queued'
                            ? 'bg-yellow-500/10 text-yellow-400 ring-1 ring-inset ring-yellow-500/20'
                            : 'bg-slate-500/10 text-slate-400 ring-1 ring-inset ring-slate-500/20'
                        }`}>
                          {song.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-center py-16 animate-fade-in">
                <p className="mb-4 text-slate-400">No songs yet. Create your first AI-generated song.</p>
                <Button asChild className="bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors">
                  <Link href="/songs/new">Create Song</Link>
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
