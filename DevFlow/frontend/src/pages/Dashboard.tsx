import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DashboardMetrics, mockDashboard, mockProjects, type ProjectList } from "@/lib/api";
import { MetricCard } from "@/components/MetricCard";

export function Dashboard() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [proj, setProj] = useState<ProjectList | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [load, setLoad] = useState(true);

  useEffect(() => {
    let c = true;
    (async () => {
      try {
        setErr(null);
        const [m, p] = await Promise.all([api.dashboard.metrics(), api.projects.list()]);
        if (c) {
          setMetrics(m);
          setProj(p);
        }
      } catch {
        if (c) {
          setErr("Showing cached sample data. Start the API for live metrics.");
          setMetrics({ ...mockDashboard, from_metrics_events_sample: mockDashboard.from_metrics_events_sample || [] } as DashboardMetrics);
          setProj(mockProjects);
        }
      } finally {
        if (c) setLoad(false);
      }
    })();
    return () => {
      c = false;
    };
  }, []);

  const s = metrics?.from_metrics_state;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="df-h1">Control center</h1>
        <p className="mt-1 text-sm text-slate-400">Simulated releases, canaries, and AI-assisted postmortems in one place.</p>
        {err && <p className="mt-2 text-sm text-amber-300/90">{err}</p>}
      </div>
      {load && <p className="text-sm text-slate-500">Loading…</p>}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Pipelines (success)" value={s?.pipeline_success_total ?? "—"} hint="Cumulative simulated successes" />
        <MetricCard title="Pipelines (failed)" value={s?.pipeline_failure_total ?? "—"} hint="Includes sample deterministic failures" />
        <MetricCard title="API requests" value={s?.api_request_count ?? "—"} hint="From middleware" />
        <MetricCard
          title="Avg. pipeline duration (s)"
          value={s?.average_pipeline_duration_seconds != null ? s.average_pipeline_duration_seconds.toFixed(1) : "—"}
        />
      </div>
      <div className="df-card">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-lg font-medium text-white">Recent projects</h2>
          <Link className="text-sm text-brand-200" to="/projects">
            Manage
          </Link>
        </div>
        <ul className="divide-y divide-slate-800/80 text-sm">
          {proj?.items.length ? (
            proj.items.map((p) => (
              <li key={p.id} className="flex justify-between py-2">
                <span className="font-medium text-slate-200">{p.name}</span>
                <span className="text-slate-500">{p.slug}</span>
              </li>
            ))
          ) : (
            <li className="py-2 text-slate-500">No data yet. Create a project in Projects.</li>
          )}
        </ul>
      </div>
    </div>
  );
}
