import { type ReactNode } from "react";

type Props = { title: string; value: ReactNode; hint?: string; accent?: string };

export function MetricCard({ title, value, hint, accent = "from-brand-500/30 to-slate-900/0" }: Props) {
  return (
    <div
      className={`df-card relative overflow-hidden border-slate-800/80 bg-gradient-to-br ${accent}`}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{title}</p>
      <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}
