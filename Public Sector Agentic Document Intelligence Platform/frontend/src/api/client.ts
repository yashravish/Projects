import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';
import { z } from 'zod';

import { AuthRefreshResponseSchema } from './schemas';

const ACCESS_TOKEN_KEY = 'psdi.access_token';
const REFRESH_TOKEN_KEY = 'psdi.refresh_token';

/* Token store — kept out of React state so axios interceptors can reach it. */
export const tokenStore = {
  getAccess(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  },
  getRefresh(): string | null {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  },
  set(access: string, refresh: string): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, access);
    localStorage.setItem(REFRESH_TOKEN_KEY, refresh);
  },
  setAccess(access: string): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, access);
  },
  clear(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  },
};

/* `vite` proxies `/api` and `/health` to the backend in dev. In a production
   nginx build, the same paths are proxied at the edge. We intentionally do
   not embed an absolute URL — the frontend never speaks to a different
   origin than the page it loads from. */
const baseURL = '/';

let isRefreshing = false;
let pendingRequests: Array<(token: string | null) => void> = [];

function onRefreshed(token: string | null): void {
  for (const cb of pendingRequests) cb(token);
  pendingRequests = [];
}

export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function extractError(err: AxiosError): ApiError {
  const status = err.response?.status ?? 0;
  const data = err.response?.data as Record<string, unknown> | undefined;
  let message = err.message;
  let code: string | undefined;
  if (data?.detail) {
    const detail = data.detail;
    if (typeof detail === 'string') {
      message = detail;
    } else if (typeof detail === 'object' && detail !== null) {
      const d = detail as { code?: string; message?: string };
      code = d.code;
      message = d.message ?? message;
    }
  }
  return new ApiError(message, status, code);
}

export const api: AxiosInstance = axios.create({ baseURL, timeout: 30_000 });

/** Query params shared by paginated `GET` list endpoints (`page`, `page_size`). */
export function paginationParams(
  page: number,
  pageSize: number,
): Record<string, number> {
  return { page, page_size: pageSize };
}

api.interceptors.request.use((config) => {
  const token = tokenStore.getAccess();
  if (token) {
    config.headers = config.headers ?? {};
    (config.headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as (AxiosRequestConfig & { _retry?: boolean }) | undefined;
    const status = error.response?.status;
    const url = original?.url ?? '';

    /* If a 401 comes back on a non-auth route, attempt one silent refresh. */
    if (status === 401 && original && !original._retry && !url.startsWith('/api/v1/auth/')) {
      original._retry = true;
      const refresh = tokenStore.getRefresh();
      if (!refresh) {
        tokenStore.clear();
        return Promise.reject(extractError(error));
      }
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          pendingRequests.push((token) => {
            if (!token) {
              reject(extractError(error));
              return;
            }
            original.headers = original.headers ?? {};
            (original.headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
            resolve(api(original));
          });
        });
      }
      isRefreshing = true;
      try {
        const resp = await axios.post('/api/v1/auth/refresh', { refresh_token: refresh });
        const fresh = AuthRefreshResponseSchema.parse(resp.data);
        tokenStore.setAccess(fresh.access_token);
        onRefreshed(fresh.access_token);
        original.headers = original.headers ?? {};
        (original.headers as Record<string, string>)['Authorization'] = `Bearer ${fresh.access_token}`;
        return api(original);
      } catch (refreshErr) {
        onRefreshed(null);
        tokenStore.clear();
        return Promise.reject(extractError(refreshErr as AxiosError));
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(extractError(error));
  },
);

/* Typed request helper that validates the response against a Zod schema. */
export async function request<T extends z.ZodTypeAny>(
  schema: T,
  config: AxiosRequestConfig,
): Promise<z.infer<T>> {
  const resp = await api.request(config);
  const parsed = schema.safeParse(resp.data);
  if (!parsed.success) {
    throw new ApiError(
      `response did not match schema: ${parsed.error.issues.map((i) => i.path.join('.')).join(', ')}`,
      0,
      'SCHEMA_MISMATCH',
    );
  }
  return parsed.data;
}
