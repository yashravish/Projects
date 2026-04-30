import { screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderWithRouter } from '../../test-utils/render';
import TaskDetailPage from '../TaskDetailPage';
import type { Task } from '../../types/task';

const MANAGER_AUTH = {
  user: { id: 1, name: 'Manager One', email: 'm@x.com', role: 'manager' as const },
  loading: false,
  error: null,
  login: vi.fn(),
  logout: vi.fn(),
};
const ANALYST_AUTH = {
  user: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' as const },
  loading: false,
  error: null,
  login: vi.fn(),
  logout: vi.fn(),
};

let currentRole: 'manager' | 'analyst' = 'manager';

vi.mock('../../auth/AuthContext', () => ({
  useAuth: () => (currentRole === 'manager' ? MANAGER_AUTH : ANALYST_AUTH),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>(
    'react-router-dom'
  );
  return {
    ...actual,
    useParams: () => ({ id: '7' }),
    useNavigate: () => vi.fn(),
  };
});

vi.mock('../../api/tasks', () => ({
  getTask: vi.fn(),
  getWorkflowEvents: vi.fn(),
  approveTask: vi.fn(),
  rejectTask: vi.fn(),
  submitTask: vi.fn(),
  completeTask: vi.fn(),
  deleteTask: vi.fn(),
}));

import { getTask, getWorkflowEvents } from '../../api/tasks';

const submittedTask: Task = {
  id: 7,
  title: 'Manager review me',
  description: 'Test description',
  status: 'submitted',
  priority: 'high',
  assigned_to_user_id: 2,
  created_by_user_id: 2,
  assignee: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
  creator: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
  created_at: '2026-04-30T00:00:00Z',
  updated_at: '2026-04-30T00:00:00Z',
};

describe('TaskDetailPage role-based buttons', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getTask).mockResolvedValue(submittedTask);
    vi.mocked(getWorkflowEvents).mockResolvedValue([]);
  });

  it('shows approve and reject buttons to a manager on a submitted task', async () => {
    currentRole = 'manager';
    renderWithRouter(<TaskDetailPage />, { initialPath: '/tasks/7' });
    await waitFor(() => {
      expect(screen.getByText('Manager review me')).toBeInTheDocument();
    });
    expect(screen.getByTestId('approve-task')).toBeInTheDocument();
    expect(screen.getByTestId('reject-task')).toBeInTheDocument();
  });

  it('does not show approve/reject buttons to an analyst', async () => {
    currentRole = 'analyst';
    renderWithRouter(<TaskDetailPage />, { initialPath: '/tasks/7' });
    await waitFor(() => {
      expect(screen.getByText('Manager review me')).toBeInTheDocument();
    });
    expect(screen.queryByTestId('approve-task')).not.toBeInTheDocument();
    expect(screen.queryByTestId('reject-task')).not.toBeInTheDocument();
  });
});
