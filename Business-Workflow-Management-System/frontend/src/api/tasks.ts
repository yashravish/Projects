import type {
  CreateTaskInput,
  Task,
  TaskPriority,
  TaskStatus,
  UpdateTaskInput,
  WorkflowEvent,
} from '../types/task';
import { apiRequest } from './client';

export interface ListTasksParams {
  status?: TaskStatus;
  priority?: TaskPriority;
  assigned_to_user_id?: number;
}

export function listTasks(params: ListTasksParams = {}): Promise<Task[]> {
  return apiRequest<Task[]>('/tasks', { query: { ...params } });
}

export function getTask(id: number): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}`);
}

export function createTask(input: CreateTaskInput): Promise<Task> {
  return apiRequest<Task>('/tasks', { method: 'POST', body: input });
}

export function updateTask(id: number, input: UpdateTaskInput): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}`, { method: 'PUT', body: input });
}

export function deleteTask(id: number): Promise<void> {
  return apiRequest<void>(`/tasks/${id}`, { method: 'DELETE' });
}

export function submitTask(id: number, comment?: string): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}/submit`, {
    method: 'POST',
    body: comment ? { comment } : undefined,
  });
}

export function approveTask(id: number, comment?: string): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}/approve`, {
    method: 'POST',
    body: comment ? { comment } : undefined,
  });
}

export function rejectTask(id: number, comment?: string): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}/reject`, {
    method: 'POST',
    body: comment ? { comment } : undefined,
  });
}

export function completeTask(id: number, comment?: string): Promise<Task> {
  return apiRequest<Task>(`/tasks/${id}/complete`, {
    method: 'POST',
    body: comment ? { comment } : undefined,
  });
}

export function getWorkflowEvents(id: number): Promise<WorkflowEvent[]> {
  return apiRequest<WorkflowEvent[]>(`/tasks/${id}/workflow-events`);
}
