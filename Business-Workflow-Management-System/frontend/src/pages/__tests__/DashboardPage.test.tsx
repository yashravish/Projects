import { screen } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderWithRouter } from '../../test-utils/render';
import DashboardPage from '../DashboardPage';

const FAKE_USER = {
  id: 1,
  name: 'Anna Analyst',
  email: 'analyst@x.com',
  role: 'analyst',
} as const;
const FAKE_AUTH = {
  user: FAKE_USER,
  loading: false,
  error: null,
  login: vi.fn(),
  logout: vi.fn(),
};

vi.mock('../../auth/AuthContext', () => ({
  useAuth: () => FAKE_AUTH,
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('../../api/tasks', () => ({
  listTasks: vi.fn(),
}));
vi.mock('../../api/reports', () => ({
  getTasksByStatus: vi.fn(),
  getTasksByPriority: vi.fn(),
}));
vi.mock('../../api/auditLogs', () => ({
  listAuditLogs: vi.fn(),
}));

import { listTasks } from '../../api/tasks';
import { getTasksByPriority, getTasksByStatus } from '../../api/reports';
import { listAuditLogs } from '../../api/auditLogs';

describe('DashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(listTasks).mockResolvedValue([]);
    vi.mocked(getTasksByStatus).mockResolvedValue({
      total: 5,
      items: [
        { status: 'pending', count: 1 },
        { status: 'in_progress', count: 1 },
        { status: 'submitted', count: 1 },
        { status: 'approved', count: 1 },
        { status: 'rejected', count: 0 },
        { status: 'completed', count: 1 },
      ],
    });
    vi.mocked(getTasksByPriority).mockResolvedValue({
      total: 5,
      items: [
        { priority: 'low', count: 1 },
        { priority: 'medium', count: 2 },
        { priority: 'high', count: 2 },
      ],
    });
    vi.mocked(listAuditLogs).mockResolvedValue([]);
  });

  it('renders summary cards using report data', async () => {
    renderWithRouter(<DashboardPage />);
    const totalCard = await screen.findByTestId('metric-total', undefined, {
      timeout: 3000,
    });
    expect(totalCard).toHaveTextContent('5');
    expect(screen.getByTestId('summary-cards')).toBeInTheDocument();
    expect(screen.getByText(/awaiting approval/i)).toBeInTheDocument();
    expect(screen.getByText(/assigned to me/i)).toBeInTheDocument();
  });
});
