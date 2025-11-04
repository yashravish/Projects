"use client";
import { useEffect, useState, useCallback } from "react";
import { useSession } from "next-auth/react";
import type { Run } from "@/lib/types";

type SortField = "created_at" | "latency_ms" | "cost_usd" | "tokens_total";
type SortDirection = "asc" | "desc";
type FeatureFilter = "all" | "playground" | "web" | "support";

type Props = {
  refreshKey?: number;
  onLoadRun?: (run: { model: string; prompt: string; params: string }) => void;
  defaultFilter?: FeatureFilter;
};

interface ParsedParams {
  feature?: "playground" | "web" | "support";
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  numSources?: number;
  contextChunkCount?: number;
  [key: string]: any;
}

export default function RunsTable({ refreshKey = 0, onLoadRun, defaultFilter = "all" }: Props) {
  const { data: session } = useSession();
  const [rows, setRows] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [limit, setLimit] = useState<number>(10);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [featureFilter, setFeatureFilter] = useState<FeatureFilter>(defaultFilter);

  const load = useCallback(async () => {
    // Don't attempt to load runs if not authenticated
    if (!session) {
      setLoading(false);
      setRows([]);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const r = await fetch("/api/runs");
      if (!r.ok) {
        const msg = await r.text().catch(() => `Error ${r.status}`);
        setError(msg || `Error ${r.status}`);
        return;
      }
      const data = await r.json();
      setRows(data);
    } catch (e) {
      setError("Failed to load runs.");
      console.error("Failed to load runs:", e);
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    load();
  }, [refreshKey, load]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const handleClearAll = async () => {
    // Don't attempt to clear if not authenticated
    if (!session) {
      setError("Please sign in to clear runs.");
      return;
    }

    try {
      const response = await fetch("/api/runs", { method: "DELETE" });
      if (!response.ok) {
        setError("Failed to clear runs");
        return;
      }
      setRows([]);
      setShowClearConfirm(false);
    } catch (e) {
      setError("Failed to clear runs");
      console.error("Failed to clear runs:", e);
    }
  };

  const parseParams = (paramsStr: string): ParsedParams => {
    try {
      return JSON.parse(paramsStr);
    } catch {
      return {};
    }
  };

  const getFeature = (run: Run): "playground" | "web" | "support" => {
    const params = parseParams(run.params);
    if (params.feature === "web" || params.feature === "support" || params.feature === "playground") {
      return params.feature;
    }
    // Heuristics for older runs that weren't tagged
    if (typeof params.numSources === "number") return "web";
    if (typeof params.contextChunkCount === "number") return "support";
    return "playground";
  };

  const getFilteredRows = () => {
    if (featureFilter === "all") {
      return rows;
    }
    return rows.filter((r) => getFeature(r) === featureFilter);
  };

  const getSortedAndLimitedRows = () => {
    const filtered = getFilteredRows();
    const sorted = [...filtered].sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      if (sortField === "created_at") {
        aVal = new Date(a.created_at).getTime();
        bVal = new Date(b.created_at).getTime();
      } else if (sortField === "tokens_total") {
        aVal = a.tokens_input + a.tokens_output;
        bVal = b.tokens_input + b.tokens_output;
      } else {
        aVal = a[sortField];
        bVal = b[sortField];
      }

      if (sortDirection === "asc") {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return sorted.slice(0, limit);
  };

  const FeatureBadge = ({ feature }: { feature: "playground" | "web" | "support" }) => {
    const colors = {
      playground: "bg-blue-500/20 text-blue-300 border-blue-500/30",
      web: "bg-purple-500/20 text-purple-300 border-purple-500/30",
      support: "bg-green-500/20 text-green-300 border-green-500/30",
    };

    const labels = {
      playground: "Playground",
      web: "Web",
      support: "Support",
    };

    return (
      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${colors[feature]}`}>
        {labels[feature]}
      </span>
    );
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <span className="text-gray-300">↕</span>;
    return sortDirection === "asc" ? <span>↑</span> : <span>↓</span>;
  };

  if (loading) {
    return (
      <div className="text-sm text-gray-500" role="status" aria-live="polite">
        Loading runs...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2" role="alert">
        {error}
      </div>
    );
  }

  if (!rows.length) return null;

  const displayRows = getSortedAndLimitedRows();
  const filteredRows = getFilteredRows();

  return (
    <div className="glass-strong rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm">Recent runs</h2>
        <div className="flex items-center gap-3">
          <label htmlFor="feature-filter" className="text-xs text-slate-400">Filter</label>
          <select
            id="feature-filter"
            value={featureFilter}
            onChange={(e) => setFeatureFilter(e.target.value as FeatureFilter)}
            className="glass border border-slate-600/30 rounded px-2 py-1 text-xs bg-transparent"
            aria-label="Filter by feature"
          >
            <option value="all" className="bg-slate-800">All</option>
            <option value="playground" className="bg-slate-800">Playground</option>
            <option value="web" className="bg-slate-800">Web</option>
            <option value="support" className="bg-slate-800">Support</option>
          </select>
          <label htmlFor="limit-select" className="text-xs text-slate-400">Show</label>
          <select
            id="limit-select"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="glass border border-slate-600/30 rounded px-2 py-1 text-xs bg-transparent"
            aria-label="Number of runs to display"
          >
            <option value={10} className="bg-slate-800">10</option>
            <option value={25} className="bg-slate-800">25</option>
            <option value={50} className="bg-slate-800">50</option>
            <option value={100} className="bg-slate-800">All</option>
          </select>
          <button
            onClick={() => setShowClearConfirm(true)}
            className="text-xs text-red-400 hover:text-red-300"
            aria-label="Clear all runs"
          >
            Clear all
          </button>
        </div>
      </div>

      {showClearConfirm && (
        <div className="mb-4 p-3 glass-strong border border-slate-700/40 rounded" role="alertdialog" aria-labelledby="clear-confirm-title">
          <p id="clear-confirm-title" className="text-sm mb-2">Delete all {rows.length} run{rows.length !== 1 ? "s" : ""}?</p>
          <div className="flex gap-2">
            <button onClick={handleClearAll} className="px-3 py-1.5 rounded bg-red-600 text-white text-xs">Delete</button>
            <button onClick={() => setShowClearConfirm(false)} className="px-3 py-1.5 rounded text-xs glass border border-slate-600/40">Cancel</button>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b border-slate-700/50">
              <th className="py-3 pr-4">
                <button
                  onClick={() => handleSort("created_at")}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
                  aria-label="Sort by date"
                >
                  When <SortIcon field="created_at" />
                </button>
              </th>
              <th className="py-3 pr-4 text-xs text-slate-400">Feature</th>
              <th className="py-3 pr-4 text-xs text-slate-400">Model</th>
              <th className="py-3 pr-4">
                <button
                  onClick={() => handleSort("tokens_total")}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
                  aria-label="Sort by tokens"
                >
                  Tokens <SortIcon field="tokens_total" />
                </button>
              </th>
              <th className="py-3 pr-4">
                <button
                  onClick={() => handleSort("latency_ms")}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
                  aria-label="Sort by latency"
                >
                  Latency <SortIcon field="latency_ms" />
                </button>
              </th>
              <th className="py-3 pr-4">
                <button
                  onClick={() => handleSort("cost_usd")}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
                  aria-label="Sort by cost"
                >
                  Cost <SortIcon field="cost_usd" />
                </button>
              </th>
              <th className="py-3 pr-4 text-xs text-slate-400">Details</th>
            </tr>
          </thead>
          <tbody>
            {displayRows.map((r, idx) => {
              const feature = getFeature(r);
              const params = parseParams(r.params);

              return (
                <tr
                  key={r.id}
                  className={`border-b border-slate-800/30 transition-colors group ${
                    onLoadRun
                      ? "hover:bg-white/5 cursor-pointer"
                      : ""
                  }`}
                  onClick={() => onLoadRun?.(r)}
                  title={onLoadRun ? "Click to load this configuration" : undefined}
                  style={{ animationDelay: `${idx * 50}ms` }}
                >
                  <td className="py-3 pr-4 text-slate-300 text-xs">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="py-3 pr-4">
                    <FeatureBadge feature={feature} />
                  </td>
                  <td className="py-3 pr-4">
                    <span className="px-2 py-1 glass border border-slate-600/30 rounded-md text-xs font-mono text-slate-200">
                      {r.model}
                    </span>
                  </td>
                  <td className="py-3 pr-4">
                    <span className="font-mono text-sm text-slate-300">
                      <span>{r.tokens_input}</span>
                      <span className="text-slate-600 mx-1">/</span>
                      <span>{r.tokens_output}</span>
                    </span>
                  </td>
                  <td className="py-3 pr-4 font-mono text-sm text-slate-300">{r.latency_ms}<span className="text-xs text-slate-500 ml-0.5">ms</span></td>
                  <td className="py-3 pr-4 font-mono text-sm text-slate-200">${r.cost_usd.toFixed(5)}</td>
                  <td className="py-3 pr-4">
                    {feature === "web" && params.numSources && (
                      <span className="text-xs text-slate-400">
                        {params.numSources} sources
                      </span>
                    )}
                    {feature === "support" && params.contextChunkCount && (
                      <span className="text-xs text-slate-400">
                        {params.contextChunkCount} chunks
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {filteredRows.length > limit && (
        <div className="mt-3 text-center text-xs text-slate-400">
          Showing {displayRows.length} of {filteredRows.length} {featureFilter !== "all" ? `${featureFilter} ` : ""}runs
        </div>
      )}
      {featureFilter !== "all" && filteredRows.length !== rows.length && (
        <div className="mt-2 text-center text-xs text-slate-500">
          ({rows.length} total runs)
        </div>
      )}
    </div>
  );
}
