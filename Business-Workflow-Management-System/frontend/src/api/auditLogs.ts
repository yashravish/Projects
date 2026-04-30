import type { AuditLog } from '../types/auditLog';
import { apiRequest } from './client';

export interface ListAuditLogsParams {
  action?: string;
  limit?: number;
}

export function listAuditLogs(params: ListAuditLogsParams = {}): Promise<AuditLog[]> {
  return apiRequest<AuditLog[]>('/audit-logs', { query: { ...params } });
}
