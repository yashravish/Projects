import { screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderWithRouter } from '../../test-utils/render';
import TaskListPage from '../TaskListPage';

vi.mock('../../api/tasks', () => ({
  listTasks: vi.fn(),
}));

import { listTasks } from '../../api/tasks';

describe('TaskListPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders fetched tasks in a table', async () => {
    vi.mocked(listTasks).mockResolvedValue([
      {
        id: 1,
        title: 'Reconcile invoices',
        description: '',
        status: 'pending',
        priority: 'high',
        assigned_to_user_id: 2,
        created_by_user_id: 2,
        assignee: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
        creator: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
        created_at: '2026-04-29T10:00:00Z',
        updated_at: '2026-04-29T10:00:00Z',
      },
      {
        id: 2,
        title: 'Vendor risk review',
        description: '',
        status: 'submitted',
        priority: 'medium',
        assigned_to_user_id: 3,
        created_by_user_id: 2,
        assignee: { id: 3, name: 'Brian Carter', email: 'b@x.com', role: 'analyst' },
        creator: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
        created_at: '2026-04-28T10:00:00Z',
        updated_at: '2026-04-28T10:00:00Z',
      },
    ]);

    renderWithRouter(<TaskListPage />, { initialPath: '/tasks' });

    await waitFor(() => {
      expect(screen.getByText('Reconcile invoices')).toBeInTheDocument();
      expect(screen.getByText('Vendor risk review')).toBeInTheDocument();
    });
    expect(screen.getAllByTestId('task-row')).toHaveLength(2);
  });

  it('shows error message when API fails', async () => {
    vi.mocked(listTasks).mockRejectedValue(new Error('Network down'));
    renderWithRouter(<TaskListPage />, { initialPath: '/tasks' });
    expect(await screen.findByRole('alert')).toHaveTextContent(/network down/i);
  });
});
