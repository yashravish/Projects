import { useEffect, useState } from "react";
import { api, type Deployment, type Project, mockProjects } from "@/lib/api";

export function Deployments() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState(1);
  const [rows, setRows] = useState<Deployment[]>([]);
  const [version, setVersion] = useState("v0.0.0-local");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadP = async () => {
    try {
      const p = await api.projects.list();
      setProjects(p.items);
      if (p.items[0]) setProjectId(p.items[0].id);
    } catch {
      setProjects(mockProjects.items);
    }
  };
  const loadD = async (pid: number) => {
    try {
      setErr(null);
      const d = await api.deployments.byProject(pid);
      setRows(d);
    } catch {
      setRows([]);
    }
  };
  useEffect(() => {
    void loadP();
  }, []);
  useEffect(() => {
    if (projectId) void loadD(projectId);
  }, [projectId]);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Deployments</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded-lg border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          value={projectId}
          onChange={(e) => setProjectId(+e.target.value)}
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        <input
          className="rounded-lg border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
        <button
          type="button"
          className="df-btn"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              setErr(null);
              await api.deployments.create(projectId, { version, canary: true, canary_start_percent: 25 });
              await loadD(projectId);
            } catch (e) {
              setErr(String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          Deploy (canary 25%)
        </button>
        <button
          type="button"
          className="df-btn df-btn-primary"
          disabled={busy || !rows[0]}
          onClick={async () => {
            setBusy(true);
            try {
              if (!rows[0]) return;
              await api.deployments.canary(rows[0].id, 100);
              await loadD(projectId);
            } catch (e) {
              setErr(String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          Full rollout to 100% (newest)
        </button>
      </div>
      <div className="df-card overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Id</th>
              <th className="pb-2">Version</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Canary %</th>
              <th className="pb-2">Error rate</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {rows.map((d) => (
              <tr key={d.id}>
                <td className="py-2">{d.id}</td>
                <td className="py-2 text-slate-200">{d.version}</td>
                <td className="py-2 text-amber-100">{d.status}</td>
                <td className="py-2 text-slate-300">{d.canary_percent}</td>
                <td className="py-2 text-slate-400">{d.error_rate}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <p className="text-sm text-slate-500">No deployments yet in this project.</p>}
      </div>
    </div>
  );
}
