import { useEffect, useState } from "react";
import { api, type FeatureFlag } from "@/lib/api";

const mockFlags: FeatureFlag[] = [
  { id: 1, name: "dark_mode_ui", description: "Sample", enabled: true, rollout_percentage: 35, environment: "default" },
];

export function Flags() {
  const [rows, setRows] = useState<FeatureFlag[]>([]);
  const [form, setForm] = useState({ name: "", description: "", enabled: true, rollout_percentage: 20, environment: "default" });
  const [err, setErr] = useState<string | null>(null);
  const [evalU, setEvalU] = useState("user-123");

  const load = async () => {
    try {
      setErr(null);
      setRows(await api.flags.list());
    } catch {
      setErr("Using sample data.");
      setRows(mockFlags);
    }
  };
  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Feature flags</h1>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <div className="df-card max-w-md space-y-2">
        <h2 className="text-sm text-slate-300">New flag</h2>
        <input
          className="w-full rounded border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          placeholder="name"
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
        <input
          className="w-full rounded border border-slate-700/80 bg-slate-900/50 px-2 py-1 text-sm"
          placeholder="description"
          value={form.description}
          onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
        />
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={form.enabled} onChange={(e) => setForm((f) => ({ ...f, enabled: e.target.checked }))} />
          enabled
        </label>
        <div className="text-sm text-slate-300">
          rollout %{" "}
          <input
            type="number"
            className="w-20 rounded border border-slate-700/80 bg-slate-900/50 px-1"
            value={form.rollout_percentage}
            onChange={(e) => setForm((f) => ({ ...f, rollout_percentage: +e.target.value }))}
          />
        </div>
        <button
          className="df-btn df-btn-primary w-full"
          onClick={async () => {
            try {
              setErr(null);
              await api.flags.create({ ...form });
              setForm({ name: "", description: "", enabled: true, rollout_percentage: 20, environment: "default" });
              await load();
            } catch (e) {
              setErr(String(e));
            }
          }}
        >
          Create flag
        </button>
      </div>
      <div className="df-card">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-400">
              <th className="pb-2">Name</th>
              <th className="pb-2">Enabled</th>
              <th className="pb-2">Rollout</th>
              <th className="pb-2">Env</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80 text-slate-200">
            {rows.map((f) => (
              <tr key={f.id}>
                <td className="py-2 pr-2 font-medium">{f.name}</td>
                <td className="py-2">{f.enabled ? "yes" : "no"}</td>
                <td className="py-2">{f.rollout_percentage}%</td>
                <td className="py-2 text-slate-500">{f.environment}</td>
                <td className="py-2">
                  <button
                    className="df-btn !py-0.5 !text-xs"
                    onClick={async () => {
                      const r = await api.flags.evaluate(f.id, evalU);
                      alert(`Granted: ${r.granted}, bucket: ${r.user_bucket}, rollout: ${r.rollout}`);
                    }}
                    type="button"
                  >
                    Test user
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-2 flex items-center gap-2 text-sm text-slate-400">
          <span>user id for test:</span>
          <input
            className="rounded border border-slate-700/80 bg-slate-900/50 px-2 py-0.5"
            value={evalU}
            onChange={(e) => setEvalU(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}
