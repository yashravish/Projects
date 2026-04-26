import { useEffect, useState } from "react";
import { api, type Defect, type DefectStats } from "@/lib/api";

const mockStats: DefectStats = { open: 3, resolved: 12, defect_rate: 0.12, by_severity: { high: 1, medium: 2 } };
const mockDefects: Defect[] = [
  { id: 1, project_id: 1, title: "Connection refused in auth", severity: "high", status: "open", priority: "p1" },
];

export function Defects() {
  const [stats, setStats] = useState<DefectStats | null>(null);
  const [rows, setRows] = useState<Defect[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        const [s, d] = await Promise.all([api.defects.stats(), api.defects.list()]);
        setStats(s);
        setRows(d);
      } catch {
        setErr("Using sample data");
        setStats(mockStats);
        setRows(mockDefects);
      }
    })();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Defects</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="df-card">
            <p className="text-xs text-slate-400">Open</p>
            <p className="text-2xl font-semibold text-white">{stats.open}</p>
          </div>
          <div className="df-card">
            <p className="text-xs text-slate-400">Resolved</p>
            <p className="text-2xl font-semibold text-white">{stats.resolved}</p>
          </div>
          <div className="df-card">
            <p className="text-xs text-slate-400">Defect rate</p>
            <p className="text-2xl font-semibold text-white">{stats.defect_rate}</p>
          </div>
          <div className="df-card">
            <p className="text-xs text-slate-400">By severity</p>
            <p className="text-sm text-slate-200">
              {Object.entries(stats.by_severity)
                .map(([k, v]) => `${k}: ${v}`)
                .join(" · ")}
            </p>
          </div>
        </div>
      )}
      <div className="df-card overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Title</th>
              <th className="pb-2">Severity</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Priority</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {rows.map((d) => (
              <tr key={d.id}>
                <td className="py-2 pr-2 text-slate-100">{d.title}</td>
                <td className="py-2 text-amber-100">{d.severity}</td>
                <td className="py-2 text-slate-300">{d.status}</td>
                <td className="py-2 text-slate-400">{d.priority}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
