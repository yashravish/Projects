import { useEffect, useState } from "react";
import { api, type ABExperiment, type Project, mockProjects } from "@/lib/api";

export function Experiments() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState(1);
  const [rows, setRows] = useState<ABExperiment[]>([]);
  const [form, setForm] = useState({ key: "new_experiment", name: "Sample", traffic_a_percent: 50 });
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    try {
      const p = await api.projects.list();
      setProjects(p.items);
      if (p.items[0]) setProjectId(p.items[0].id);
    } catch {
      setProjects(mockProjects.items);
    }
  };
  const loadE = async (pid: number) => {
    try {
      setErr(null);
      setRows(await api.experiments.byProject(pid));
    } catch {
      setRows([]);
    }
  };
  useEffect(() => {
    void load();
  }, []);
  useEffect(() => {
    if (projectId) void loadE(projectId);
  }, [projectId]);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">A/B experiments</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <div className="flex flex-wrap items-end gap-2">
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
          className="rounded border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          value={form.key}
          onChange={(e) => setForm((f) => ({ ...f, key: e.target.value }))}
        />
        <input
          className="rounded border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
        <button
          className="df-btn df-btn-primary"
          onClick={async () => {
            try {
              setErr(null);
              await api.experiments.create({
                project_id: projectId,
                key: form.key,
                name: form.name,
                traffic_a_percent: form.traffic_a_percent,
              });
              await loadE(projectId);
            } catch (e) {
              setErr(String(e));
            }
          }}
          type="button"
        >
          Create experiment
        </button>
      </div>
      <div className="df-card overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Key</th>
              <th className="pb-2">Name</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Traffic A %</th>
              <th className="pb-2">Assign</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {rows.map((e) => (
              <tr key={e.id}>
                <td className="py-2 font-mono text-xs text-slate-200">{e.key}</td>
                <td className="py-2 text-slate-200">{e.name}</td>
                <td className="py-2 text-amber-100">{e.status}</td>
                <td className="py-2 text-slate-300">{e.traffic_a_percent}</td>
                <td className="py-2">
                  <button
                    className="df-btn !py-0.5 !text-xs"
                    type="button"
                    onClick={async () => {
                      const r = await api.experiments.assign(e.id, "demo-user");
                      alert(`Variant: ${r.variant} (${r.variant_name})`);
                    }}
                  >
                    Assign demo user
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <p className="py-4 text-sm text-slate-500">No experiments in this project.</p>}
      </div>
    </div>
  );
}
