import type { UserSummary } from './auth';

export const AUDIT_ACTIONS = [
  'login',
  'create_task',
  'update_task',
  'delete_task',
  'submit_task',
  'approve_task',
  'reject_task',
  'complete_task',
] as const;

export type AuditAction = (typeof AUDIT_ACTIONS)[number];

export interface AuditLog {
  id: number;
  user_id: number | null;
  user: UserSummary | null;
  action: AuditAction;
  entity_type: string;
  entity_id: number | null;
  details: string | null;
  created_at: string;
}
