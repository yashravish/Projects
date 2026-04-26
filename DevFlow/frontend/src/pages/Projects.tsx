import { useEffect, useState } from "react";
import { api, type Project, mockProjects } from "@/lib/api";

export function Projects() {
  const [rows, setRows] = useState<Project[]>([]);
  const [total, setTotal] = useState(0);
  const [form, setForm] = useState({ name: "", slug: "", description: "" });
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    try {
      setErr(null);
      const r = await api.projects.list();
      setRows(r.items);
      setTotal(r.total);
    } catch {
      setErr("Offline: showing sample data.");
      setRows(mockProjects.items);
      setTotal(mockProjects.total);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="df-h1">Projects</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <div className="df-card max-w-md space-y-3">
        <h2 className="text-sm font-medium text-slate-300">Create project</h2>
        <input
          className="w-full rounded-lg border border-slate-700/80 bg-slate-900/50 px-3 py-2 text-sm"
          placeholder="Name"
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
        <input
          className="w-full rounded-lg border border-slate-700/80 bg-slate-900/50 px-3 py-2 text-sm"
          placeholder="Slug (e.g. payments-api)"
          value={form.slug}
          onChange={(e) => setForm((f) => ({ ...f, slug: e.target.value }))}
        />
        <input
          className="w-full rounded-lg border border-slate-700/80 bg-slate-900/50 px-3 py-2 text-sm"
          placeholder="Description"
          value={form.description}
          onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
        />
        <button
          type="button"
          disabled={saving || !form.name || !form.slug}
          className="df-btn df-btn-primary w-full"
          onClick={async () => {
            setSaving(true);
            try {
              setErr(null);
              await api.projects.create({ name: form.name, slug: form.slug, description: form.description || null });
              setForm({ name: "", slug: "", description: "" });
              await load();
            } catch (e) {
              setErr(String(e));
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "Creating…" : "Create project"}
        </button>
      </div>
      <div className="df-card">
        <div className="mb-2 text-sm text-slate-500">Total: {total}</div>
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Name</th>
              <th className="pb-2">Slug</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80 text-slate-200">
            {rows.map((p) => (
              <tr key={p.id} className="align-top">
                <td className="py-2 pr-2 font-medium">{p.name}</td>
                <td className="py-2 text-slate-500">{p.slug}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
