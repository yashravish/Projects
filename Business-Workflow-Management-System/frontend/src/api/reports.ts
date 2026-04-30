import type {
  ApprovalSummaryReport,
  TasksByPriorityReport,
  TasksByStatusReport,
  UserWorkloadReport,
} from '../types/report';
import { API_BASE_URL, apiRequest, getToken } from './client';

export function getTasksByStatus(): Promise<TasksByStatusReport> {
  return apiRequest<TasksByStatusReport>('/reports/tasks-by-status');
}

export function getTasksByPriority(): Promise<TasksByPriorityReport> {
  return apiRequest<TasksByPriorityReport>('/reports/tasks-by-priority');
}

export function getUserWorkload(): Promise<UserWorkloadReport> {
  return apiRequest<UserWorkloadReport>('/reports/user-workload');
}

export function getApprovalSummary(): Promise<ApprovalSummaryReport> {
  return apiRequest<ApprovalSummaryReport>('/reports/approval-summary');
}

export async function exportTasksCsv(): Promise<Blob> {
  const token = getToken();
  const response = await fetch(`${API_BASE_URL}/reports/export/tasks.csv`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error(`CSV export failed: ${response.status}`);
  }
  return response.blob();
}
