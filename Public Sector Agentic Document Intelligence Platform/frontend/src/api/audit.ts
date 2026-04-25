import { api, paginationParams, request, tokenStore } from './client';
import {
  AuditEventListSchema,
  AuditEventSchema,
  AuditOutcomeSchema,
  IntegrityBreakSchema,
  IntegrityReportSchema,
  LedgerStatsSchema,
  RetentionPolicyListSchema,
  RetentionPolicySchema,
  RetentionResourceSchema,
  RetentionRunListSchema,
  RetentionRunSchema,
  RetentionStatusSchema,
} from './schemas';
import type {
  AuditEvent,
  AuditEventList,
  AuditOutcome,
  IntegrityBreak,
  IntegrityReport,
  LedgerStats,
  RetentionPolicy,
  RetentionPolicyList,
  RetentionResource,
  RetentionRun,
  RetentionRunList,
  RetentionStatus,
} from './schemas';

export {
  AuditEventListSchema,
  AuditEventSchema,
  AuditOutcomeSchema,
  IntegrityBreakSchema,
  IntegrityReportSchema,
  LedgerStatsSchema,
  RetentionPolicyListSchema,
  RetentionPolicySchema,
  RetentionResourceSchema,
  RetentionRunListSchema,
  RetentionRunSchema,
  RetentionStatusSchema,
};

export type {
  AuditEvent,
  AuditEventList,
  AuditOutcome,
  IntegrityBreak,
  IntegrityReport,
  LedgerStats,
  RetentionPolicy,
  RetentionPolicyList,
  RetentionResource,
  RetentionRun,
  RetentionRunList,
  RetentionStatus,
};

// Filters and request payloads

export interface AuditEventFilters {
  page?: number;
  page_size?: number;
  actions?: string[];
  resource_types?: string[];
  outcomes?: AuditOutcome[];
  actor_ids?: string[];
  since?: string | null;
  until?: string | null;
  search?: string | null;
}

function filtersToParams(
  filters: AuditEventFilters,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (filters.page) out.page = filters.page;
  if (filters.page_size) out.page_size = filters.page_size;
  if (filters.actions && filters.actions.length > 0) {
    out.actions = filters.actions;
  }
  if (filters.resource_types && filters.resource_types.length > 0) {
    out.resource_types = filters.resource_types;
  }
  if (filters.outcomes && filters.outcomes.length > 0) {
    out.outcomes = filters.outcomes;
  }
  if (filters.actor_ids && filters.actor_ids.length > 0) {
    out.actor_ids = filters.actor_ids;
  }
  if (filters.since) out.since = filters.since;
  if (filters.until) out.until = filters.until;
  if (filters.search && filters.search.trim().length > 0) {
    out.search = filters.search.trim();
  }
  return out;
}

// API surface

export async function listAuditEvents(
  filters: AuditEventFilters = {},
): Promise<AuditEventList> {
  return request(AuditEventListSchema, {
    url: '/api/v1/audit/events',
    method: 'GET',
    params: filtersToParams({ page: 1, page_size: 50, ...filters }),
    /* The backend serializes list query params as repeated `?key=v1&key=v2`,
       which is axios' default — no `paramsSerializer` override needed. */
  });
}

export async function getAuditEvent(eventId: string): Promise<AuditEvent> {
  return request(AuditEventSchema, {
    url: `/api/v1/audit/events/${eventId}`,
    method: 'GET',
  });
}

export async function getLedgerStats(): Promise<LedgerStats> {
  return request(LedgerStatsSchema, {
    url: '/api/v1/audit/stats',
    method: 'GET',
  });
}

export async function verifyIntegrity(): Promise<IntegrityReport> {
  return request(IntegrityReportSchema, {
    url: '/api/v1/audit/integrity',
    method: 'GET',
  });
}

export async function listRetentionPolicies(): Promise<RetentionPolicyList> {
  return request(RetentionPolicyListSchema, {
    url: '/api/v1/audit/policies',
    method: 'GET',
  });
}

export interface UpsertRetentionPolicyPayload {
  ttl_days: number;
  is_active?: boolean;
  notes?: string | null;
}

export async function upsertRetentionPolicy(
  resourceType: RetentionResource,
  payload: UpsertRetentionPolicyPayload,
): Promise<RetentionPolicy> {
  return request(RetentionPolicySchema, {
    url: `/api/v1/audit/policies/${resourceType}`,
    method: 'PUT',
    data: {
      ttl_days: payload.ttl_days,
      is_active: payload.is_active ?? true,
      notes: payload.notes ?? null,
    },
  });
}

export async function listRetentionRuns(
  page = 1,
  pageSize = 25,
): Promise<RetentionRunList> {
  return request(RetentionRunListSchema, {
    url: '/api/v1/audit/retention/runs',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function runRetention(): Promise<RetentionRun> {
  return request(RetentionRunSchema, {
    url: '/api/v1/audit/retention/runs',
    method: 'POST',
  });
}

/**
 * Streaming CSV download. Bypasses the typed `request` helper because the
 * response is a binary blob, not JSON. Uses the same axios instance so the
 * bearer token and refresh-on-401 logic still apply.
 */
export async function exportEventsCsv(
  filters: AuditEventFilters = {},
): Promise<Blob> {
  const resp = await api.request({
    url: '/api/v1/audit/events.csv',
    method: 'GET',
    params: filtersToParams(filters),
    responseType: 'blob',
  });
  return resp.data as Blob;
}

/**
 * Helper for pages: download the current filter set as a CSV file. Returns
 * the suggested filename so callers can show toast feedback.
 */
export async function downloadAuditCsv(
  filters: AuditEventFilters = {},
): Promise<string> {
  const blob = await exportEventsCsv(filters);
  const stamp = new Date()
    .toISOString()
    .replace(/[:.]/g, '-')
    .replace(/Z$/, '');
  const filename = `audit-ledger-${stamp}.csv`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  /* Defer revoke until the click has actually fired. */
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return filename;
}

/* Re-export for convenience so call-sites can import a single module. */
export { tokenStore };
