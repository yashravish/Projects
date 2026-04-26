import { useEffect, useState } from "react";
import { api, type DashboardMetrics, mockDashboard } from "@/lib/api";
import { MetricCard } from "@/components/MetricCard";

export function MetricsPage() {
  const [d, setD] = useState<DashboardMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        setD(await api.dashboard.metrics());
      } catch {
        setD({ ...mockDashboard, from_metrics_events_sample: mockDashboard.from_metrics_events_sample } as DashboardMetrics);
        setErr("Using sample: API unreachable");
      }
    })();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Metrics &amp; observability</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <p className="text-sm text-slate-400">
        Prometheus:{" "}
        <a className="text-brand-200 underline" href="http://localhost:8000/metrics" target="_blank" rel="noreferrer">
          /metrics
        </a>{" "}
        (served on the API)
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <MetricCard title="Pipelines (success)" value={d?.from_metrics_state?.pipeline_success_total ?? "—"} />
        <MetricCard title="Pipelines (failed)" value={d?.from_metrics_state?.pipeline_failure_total ?? "—"} />
        <MetricCard title="Deployments (ok)" value={d?.from_metrics_state?.deployment_success_total ?? "—"} />
      </div>
      <div className="df-card text-sm text-slate-300">
        <h2 className="mb-2 text-sm font-medium text-white">Events sample</h2>
        <ul className="space-y-1 text-xs">
          {d?.from_metrics_events_sample?.map((e, i) => (
            <li key={i} className="font-mono text-slate-400">
              {e.name} = {e.value} @ {e.ts}
            </li>
          ))}
        </ul>
        {!d?.from_metrics_events_sample?.length && <p className="text-slate-500">No events in DB</p>}
      </div>
    </div>
  );
}
