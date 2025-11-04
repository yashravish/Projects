import ReplayPlayer from "@/components/ReplayPlayer";
import { prisma } from "@/lib/prisma";
import type { Metadata } from "next";

type Props = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const run = await prisma.run.findUnique({ where: { id } });
  if (!run || !run.is_public) return { title: "Replay not available" };
  return {
    title: `Replay – ${run.model}`,
    description: run.prompt.slice(0, 140),
    openGraph: {
      title: `Replay – ${run.model}`,
      description: run.prompt.slice(0, 140)
    },
    twitter: {
      card: "summary_large_image",
      title: `Replay – ${run.model}`,
      description: run.prompt.slice(0, 140)
    }
  };
}

export default async function ReplayPage({ params }: Props) {
  const { id } = await params;
  const run = await prisma.run.findUnique({ where: { id } });
  if (!run || !run.is_public) {
    return (
      <main className="min-h-screen p-6">
        <div className="max-w-3xl mx-auto">
          <div className="glass-strong rounded-xl p-6 border border-slate-700/30">
            <h1 className="text-lg font-semibold">Replay not available</h1>
            <p className="text-sm text-slate-400 mt-2">This replay is private or does not exist.</p>
          </div>
        </div>
      </main>
    );
  }

  const frames: [number, string][] = run.replay_frames ? JSON.parse(run.replay_frames) : [];
  const finalText = run.output_text ?? "";

  return (
    <main className="min-h-screen p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="glass-strong rounded-xl p-6 border border-slate-700/30">
          <h1 className="text-xl font-semibold">Run Replay</h1>
          <p className="text-sm text-slate-400 mt-2">Model: {run.model} • Tokens: {run.tokens_input + run.tokens_output} • Latency: {run.latency_ms} ms • Cost: ${run.cost_usd.toFixed(5)}</p>
        </div>

        <div className="glass-strong rounded-xl p-6 border border-slate-700/30">
          <h2 className="text-sm mb-2">Prompt</h2>
          <pre className="glass border border-slate-600/40 rounded p-3 text-sm whitespace-pre-wrap">{run.prompt}</pre>
        </div>

        <div className="glass-strong rounded-xl p-6 border border-slate-700/30">
          <h2 className="text-sm mb-3">Replay</h2>
          <ReplayPlayer frames={frames} finalText={finalText} />
        </div>
      </div>
    </main>
  );
}


