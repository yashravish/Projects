import { useState } from "react";
import { api, type AIReport } from "@/lib/api";

const sample =
  "Error: ECONNREFUSED 127.0.0.1:6379\n  at createConnection (redis/index.js)";

export function AIAnalyzer() {
  const [logs, setLogs] = useState(sample);
  const [createDef, setCreateDef] = useState(false);
  const [projectId, setProjectId] = useState(1);
  const [out, setOut] = useState<AIReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [load, setLoad] = useState(false);

  return (
    <div className="space-y-4">
      <h1 className="df-h1">AI failure analyzer</h1>
      <p className="text-sm text-slate-400">Submit CI/CD logs. Uses OpenAI when configured; otherwise a mock rule-based model.</p>
      {err && <p className="text-sm text-rose-200">{err}</p>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="df-card space-y-2">
          <label className="text-xs text-slate-400">Logs</label>
          <textarea
            className="h-64 w-full rounded-xl border border-slate-800/80 bg-slate-900/50 p-3 font-mono text-sm text-slate-100"
            value={logs}
            onChange={(e) => setLogs(e.target.value)}
          />
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-300">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={createDef} onChange={(e) => setCreateDef(e.target.checked)} />
              create defect
            </label>
            <span>project</span>
            <input
              className="w-16 rounded border border-slate-700/80 bg-slate-900/50 px-1"
              type="number"
              value={projectId}
              onChange={(e) => setProjectId(+e.target.value)}
            />
          </div>
          <button
            type="button"
            className="df-btn df-btn-primary w-full"
            disabled={load}
            onClick={async () => {
              setLoad(true);
              setErr(null);
              try {
                setOut(
                  await api.ai.analyze({
                    logs,
                    project_id: projectId,
                    create_defect: createDef,
                    link_kb_article_ids: [],
                  }),
                );
              } catch (e) {
                setErr(String(e));
                setOut({
                  id: 0,
                  root_cause_summary: "Local mock: connection refused in Redis client.",
                  likely_file_or_component: "client/redis",
                  suggested_fix: "Check Redis is running and the URL in settings.",
                  severity: "high",
                  confidence_score: 0.72,
                  created_defect_id: null,
                });
              } finally {
                setLoad(false);
              }
            }}
          >
            {load ? "Analyzing…" : "Run analysis"}
          </button>
        </div>
        <div className="df-card space-y-2">
          <h2 className="text-sm font-medium text-slate-300">Result</h2>
          {out && (
            <div className="space-y-2 text-sm text-slate-200">
              <p>
                <span className="text-slate-500">Root cause: </span>
                {out.root_cause_summary}
              </p>
              <p>
                <span className="text-slate-500">Component: </span>
                {out.likely_file_or_component}
              </p>
              <p>
                <span className="text-slate-500">Fix: </span>
                {out.suggested_fix}
              </p>
              <p>
                <span className="text-slate-500">Severity: </span>
                {out.severity} (confidence {out.confidence_score.toFixed(2)})
              </p>
              {out.created_defect_id && <p className="text-emerald-200">Defect id {out.created_defect_id} created</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
