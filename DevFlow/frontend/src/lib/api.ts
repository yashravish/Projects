const rawBase =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) || "";

/** Browser: use same-origin (Vite proxy) when empty. */
export const API_BASE: string = rawBase;

async function doFetch<T>(path: string, init?: RequestInit & { json?: object }): Promise<T> {
  const url = `${API_BASE}${path}`;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init?.json !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const r = await fetch(url, {
    ...init,
    headers: { ...headers, ...((init?.headers as Record<string, string>) || {}) },
    body: init?.json !== undefined ? JSON.stringify(init.json) : init?.body,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  if (r.status === 204) {
    return undefined as T;
  }
  return (await r.json()) as T;
}

export const api = {
  projects: {
    list: () => doFetch<ProjectList>("/api/projects"),
    create: (b: { name: string; slug: string; description?: string | null }) =>
      doFetch<Project>("/api/projects", { method: "POST", json: b }),
  },
  pipeline: {
    trigger: (projectId: number, body: { branch?: string; commit_sha?: string }) =>
      doFetch<PipelineRun>(`/api/pipelines/${projectId}/trigger`, { method: "POST", json: body }),
    byProject: (id: number) => doFetch<PipelineRun[]>(`/api/pipelines/by-project/${id}`),
  },
  deployments: {
    byProject: (id: number) => doFetch<Deployment[]>(`/api/deployments/by-project/${id}`),
    create: (projectId: number, body: { version: string; environment?: string; canary?: boolean; canary_start_percent?: number }) =>
      doFetch<Deployment>(`/api/deployments/${projectId}`, { method: "POST", json: body }),
    canary: (id: number, target_max_percent: number) =>
      doFetch<Deployment>(`/api/deployments/${id}/canary`, { method: "POST", json: { target_max_percent } }),
  },
  flags: {
    list: () => doFetch<FeatureFlag[]>("/api/flags"),
    create: (b: { name: string; description?: string; enabled: boolean; rollout_percentage: number; environment: string }) =>
      doFetch<FeatureFlag>("/api/flags", { method: "POST", json: b }),
    evaluate: (flag_id: number, user_id: string) =>
      doFetch<FlagEval>("/api/flags/evaluate", { method: "POST", json: { flag_id, user_id } }),
  },
  experiments: {
    byProject: (id: number) => doFetch<ABExperiment[]>(`/api/experiments/by-project/${id}`),
    create: (b: ABExperimentCreate) => doFetch<ABExperiment>("/api/experiments", { method: "POST", json: b }),
    assign: (experiment_id: number, user_id: string) =>
      doFetch<ABAssign>("/api/experiments/assign", { method: "POST", json: { experiment_id, user_id } }),
    metrics: (b: { experiment_id: number; variant: string; user_id: string; conversion?: boolean; latency_ms?: number; error?: boolean }) =>
      doFetch<Record<string, unknown>>("/api/experiments/metrics", { method: "POST", json: b }),
  },
  ai: {
    analyze: (b: { logs: string; project_id?: number; create_defect?: boolean; link_kb_article_ids?: number[] }) =>
      doFetch<AIReport>("/api/ai/analyze", { method: "POST", json: b }),
  },
  dashboard: {
    metrics: () => doFetch<DashboardMetrics>("/api/dashboard/metrics"),
  },
  defects: {
    list: (project_id?: number) => {
      const q = project_id != null ? `?project_id=${project_id}` : "";
      return doFetch<Defect[]>(`/api/defects${q}`);
    },
    stats: (project_id?: number) => {
      const q = project_id != null ? `?project_id=${project_id}` : "";
      return doFetch<DefectStats>(`/api/defects/stats${q}`);
    },
  },
  kb: {
    list: (q?: string) => {
      const qs = q ? `?q=${encodeURIComponent(q)}` : "";
      return doFetch<KBArticle[]>(`/api/kb${qs}`);
    },
  },
};

export type Project = {
  id: number;
  name: string;
  slug: string;
  description: string | null;
  created_at: string;
};
export type ProjectList = { items: Project[]; total: number };

export type PipelineRun = {
  id: number;
  project_id: number;
  status: string;
  branch: string;
  commit_sha: string;
  total_duration_ms: number;
  stages: { id: number; name: string; status: string; duration_ms: number; passed: boolean }[];
};

export type Deployment = {
  id: number;
  project_id: number;
  version: string;
  status: string;
  environment: string;
  canary_percent: number;
  error_rate: number;
};

export type FeatureFlag = {
  id: number;
  name: string;
  description: string | null;
  enabled: boolean;
  rollout_percentage: number;
  environment: string;
};

export type ABExperiment = {
  id: number;
  project_id: number;
  key: string;
  name: string;
  status: string;
  traffic_a_percent: number;
};

export type ABExperimentCreate = {
  project_id: number;
  key: string;
  name: string;
  traffic_a_percent?: number;
  variant_a_name?: string;
  variant_b_name?: string;
  key_metric?: string;
  notes?: string;
};

export type AIReport = {
  id: number;
  root_cause_summary: string;
  likely_file_or_component: string;
  suggested_fix: string;
  severity: string;
  confidence_score: number;
  created_defect_id: number | null;
};

export type DashboardMetrics = {
  application: string;
  from_metrics_state: {
    pipeline_success_total: number;
    pipeline_failure_total: number;
    deployment_success_total?: number;
    deployment_failure_total?: number;
    api_request_count: number;
    average_pipeline_duration_seconds: number;
  };
  from_metrics_events_sample: { name: string; value: number; ts: string }[];
};

export type Defect = {
  id: number;
  project_id: number;
  title: string;
  severity: string;
  status: string;
  priority: string;
};

export type DefectStats = { open: number; resolved: number; defect_rate: number; by_severity: Record<string, number> };

export type KBArticle = { id: number; title: string; type: string; created_at: string; slug: string };

type FlagEval = { granted: boolean; rollout: number; user_bucket: number; flag: string; environment: string };
type ABAssign = { experiment_id: number; user_id: string; variant: string; variant_name: string };

export const mockDashboard = {
  application: "devflow",
  from_metrics_state: {
    pipeline_success_total: 42,
    pipeline_failure_total: 5,
    deployment_success_total: 8,
    deployment_failure_total: 1,
    api_request_count: 1280,
    average_pipeline_duration_seconds: 38.2,
  },
  from_metrics_events_sample: [
    { name: "http.request.duration", value: 44, ts: new Date().toISOString() },
  ],
} satisfies DashboardMetrics;

export const mockProjects: ProjectList = {
  total: 1,
  items: [{ id: 1, name: "Acme", slug: "acme", description: "Offline sample", created_at: new Date().toISOString() }],
};
