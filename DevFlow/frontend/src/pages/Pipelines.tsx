import { useEffect, useState } from "react";
import { api, type PipelineRun, type Project, mockProjects } from "@/lib/api";

export function Pipelines() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState(1);
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadProjects = async () => {
    try {
      setErr(null);
      const p = await api.projects.list();
      setProjects(p.items);
      if (p.items[0]) setProjectId(p.items[0].id);
    } catch {
      setProjects(mockProjects.items);
      setProjectId(1);
      setErr("Offline: sample data.");
    }
  };

  const loadRuns = async (pid: number) => {
    try {
      setErr(null);
      const r = await api.pipeline.byProject(pid);
      setRuns(r);
    } catch {
      setRuns([]);
    }
  };

  useEffect(() => {
    void loadProjects();
  }, []);
  useEffect(() => {
    if (projectId) void loadRuns(projectId);
  }, [projectId]);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Pipeline runs</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-sm text-slate-400">Project</label>
        <select
          className="rounded-lg border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          value={projectId}
          onChange={(e) => setProjectId(+e.target.value)}
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} (#{p.id})
            </option>
          ))}
        </select>
        <button
          type="button"
          className="df-btn df-btn-primary"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              setErr(null);
              await api.pipeline.trigger(projectId, { branch: "main", commit_sha: "a1b2c3d4" });
              await loadRuns(projectId);
            } catch (e) {
              setErr(String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Running…" : "Trigger simulated pipeline"}
        </button>
      </div>
      <div className="df-card overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Id</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Branch / SHA</th>
              <th className="pb-2">Duration (ms)</th>
              <th className="pb-2">Stages</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80 text-slate-200">
            {runs.map((r) => (
              <tr key={r.id} className="align-top">
                <td className="py-2 pr-2">{r.id}</td>
                <td className="py-2 pr-2 text-emerald-200">{r.status}</td>
                <td className="py-2 pr-2 text-slate-300">
                  {r.branch} · {r.commit_sha}
                </td>
                <td className="py-2 pr-2 text-slate-400">{r.total_duration_ms}</td>
                <td className="py-2 text-xs text-slate-400">
                  {r.stages
                    .map(
                      (s) =>
                        `${s.name}=${s.status}${s.passed ? "" : " (fail)"}`,
                    )
                    .join(" → ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!runs.length && <p className="py-4 text-sm text-slate-500">No runs for this project yet.</p>}
      </div>
    </div>
  );
}
