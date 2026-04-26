import { useEffect, useState } from "react";
import { api, type KBArticle } from "@/lib/api";

const mock: KBArticle[] = [
  { id: 1, title: "Runbook: failed deploy rollback", type: "runbook", created_at: "", slug: "runbook-failed-deploy" },
];

export function Knowledge() {
  const [rows, setRows] = useState<KBArticle[]>([]);
  const [q, setQ] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        setRows(await api.kb.list(q));
      } catch {
        setErr("Sample data");
        setRows(mock);
      }
    })();
  }, [q]);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">Knowledge base</h1>
      <p className="text-sm text-slate-400">Runbooks, postmortems, and troubleshooting guides linked to incidents.</p>
      {err && <p className="text-sm text-amber-200">{err}</p>}
      <input
        className="w-full max-w-md rounded-lg border border-slate-700/80 bg-slate-900/50 px-3 py-2 text-sm"
        placeholder="Search titles…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <div className="space-y-2">
        {rows.map((a) => (
          <div key={a.id} className="df-card">
            <p className="text-xs uppercase text-slate-500">{a.type}</p>
            <p className="text-lg font-medium text-white">{a.title}</p>
            <p className="text-xs text-slate-500">{a.slug}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
