import type {
  User,
  ProcessRequest,
  RequestDetail,
  AISummary,
  AnalyticsOverview,
  CategoryCount,
  DepartmentCount,
  PriorityCount,
  StatusCount,
  PainPoint,
} from '../types';

function getToken(): string | null {
  return localStorage.getItem('token');
}

async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(url, { ...options, headers });

  if (res.status === 401) {
    localStorage.removeItem('token');
    window.location.href = '/login';
    throw new Error('Session expired. Please log in again.');
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }

  return res;
}

export async function login(
  email: string,
  password: string
): Promise<{ access_token: string; token_type: string; user: User }> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || 'Invalid credentials');
  }
  return res.json();
}

export async function getMe(): Promise<User> {
  const res = await authFetch('/api/auth/me');
  return res.json();
}

export async function getRequests(filters?: {
  department?: string;
  category?: string;
  status?: string;
}): Promise<ProcessRequest[]> {
  const params = new URLSearchParams();
  if (filters?.department) params.set('department', filters.department);
  if (filters?.category) params.set('category', filters.category);
  if (filters?.status) params.set('status', filters.status);
  params.set('limit', '200');

  const res = await authFetch(`/api/requests?${params.toString()}`);
  return res.json();
}

export async function createRequest(data: {
  title: string;
  description: string;
  category: string;
  urgency: number;
  business_impact: number;
  desired_completion_date?: string | null;
}): Promise<RequestDetail> {
  const res = await authFetch('/api/requests', {
    method: 'POST',
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function getRequest(id: number): Promise<RequestDetail> {
  const res = await authFetch(`/api/requests/${id}`);
  return res.json();
}

export async function updateRequest(
  id: number,
  data: {
    status?: string;
    assigned_owner?: string;
    note?: string;
    resolution_summary?: string;
  }
): Promise<RequestDetail> {
  const res = await authFetch(`/api/requests/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function summarizeRequest(id: number): Promise<AISummary> {
  const res = await authFetch(`/api/requests/${id}/summarize`, {
    method: 'POST',
  });
  return res.json();
}

export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  const res = await authFetch('/api/analytics/overview');
  return res.json();
}

export async function getByCategory(): Promise<CategoryCount[]> {
  const res = await authFetch('/api/analytics/by-category');
  return res.json();
}

export async function getByDepartment(): Promise<DepartmentCount[]> {
  const res = await authFetch('/api/analytics/by-department');
  return res.json();
}

export async function getByPriority(): Promise<PriorityCount[]> {
  const res = await authFetch('/api/analytics/by-priority');
  return res.json();
}

export async function getStatusDistribution(): Promise<StatusCount[]> {
  const res = await authFetch('/api/analytics/status-distribution');
  return res.json();
}

export async function getTopPainPoints(): Promise<PainPoint[]> {
  const res = await authFetch('/api/analytics/top-pain-points');
  return res.json();
}
