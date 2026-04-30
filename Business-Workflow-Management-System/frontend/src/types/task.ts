import type { UserSummary } from './auth';

export const TASK_STATUSES = [
  'pending',
  'in_progress',
  'submitted',
  'approved',
  'rejected',
  'completed',
] as const;

export type TaskStatus = (typeof TASK_STATUSES)[number];

export const TASK_PRIORITIES = ['low', 'medium', 'high'] as const;

export type TaskPriority = (typeof TASK_PRIORITIES)[number];

export type ApprovalDecision = 'approved' | 'rejected';

export interface Task {
  id: number;
  title: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  assigned_to_user_id: number;
  created_by_user_id: number;
  assignee: UserSummary;
  creator: UserSummary;
  created_at: string;
  updated_at: string;
}

export interface CreateTaskInput {
  title: string;
  description: string;
  priority: TaskPriority;
  assigned_to_user_id: number;
}

export interface UpdateTaskInput {
  title?: string;
  description?: string;
  priority?: TaskPriority;
  assigned_to_user_id?: number;
}

export interface WorkflowEvent {
  id: number;
  task_id: number;
  from_status: TaskStatus | null;
  to_status: TaskStatus;
  changed_by_user_id: number;
  changed_by: UserSummary;
  note: string | null;
  created_at: string;
}
