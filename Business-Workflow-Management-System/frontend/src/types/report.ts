import type { UserRole } from './auth';
import type { ApprovalDecision, TaskPriority, TaskStatus } from './task';

export interface StatusCount {
  status: TaskStatus;
  count: number;
}

export interface PriorityCount {
  priority: TaskPriority;
  count: number;
}

export interface UserWorkloadEntry {
  user_id: number;
  name: string;
  email: string;
  role: UserRole;
  total: number;
  pending: number;
  in_progress: number;
  submitted: number;
  approved: number;
  rejected: number;
  completed: number;
}

export interface ApprovalSummaryEntry {
  decision: ApprovalDecision;
  count: number;
}

export interface TasksByStatusReport {
  items: StatusCount[];
  total: number;
}

export interface TasksByPriorityReport {
  items: PriorityCount[];
  total: number;
}

export interface UserWorkloadReport {
  items: UserWorkloadEntry[];
}

export interface ApprovalSummaryReport {
  items: ApprovalSummaryEntry[];
  total: number;
}
