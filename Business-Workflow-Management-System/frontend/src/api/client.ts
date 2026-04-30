const DEFAULT_BASE = 'http://localhost:8000';

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ||
  DEFAULT_BASE;

const TOKEN_KEY = 'bwms.token';

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* noop */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* noop */
  }
}

export class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  query?: Record<string, unknown>;
  rawResponse?: boolean;
}

export async function apiRequest<T>(
  path: string,
  { method = 'GET', body, query, rawResponse = false }: RequestOptions = {}
): Promise<T> {
  const url = new URL(path.startsWith('http') ? path : `${API_BASE_URL}${path}`);
  if (query) {
    for (const [key, rawValue] of Object.entries(query)) {
      const value = rawValue as string | number | boolean | undefined | null;
      if (value === undefined || value === null || value === '') continue;
      url.searchParams.set(key, String(value));
    }
  }

  const headers: Record<string, string> = {
    Accept: rawResponse ? '*/*' : 'application/json',
  };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
  }
  const token = getToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(url.toString(), {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    const data: unknown = await response.json().catch(() => null);
    if (data && typeof data === 'object' && 'detail' in data) {
      const detail = (data as { detail: unknown }).detail;
      message = typeof detail === 'string' ? detail : JSON.stringify(detail);
    } else if (data === null) {
      const text = await response.text().catch(() => '');
      if (text) message = text;
    }
    const err = new ApiError(message, response.status, data ?? undefined);

    if (
      response.status === 401 &&
      !(path === '/auth/login' && method === 'POST')
    ) {
      clearToken();
      if (!window.location.pathname.startsWith('/login')) {
        window.location.assign('/login?reason=session');
      }
    }

    throw err;
  }

  if (rawResponse) {
    // @ts-expect-error - caller asks for the Response itself.
    return response;
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
