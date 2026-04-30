import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderWithRouter } from '../../test-utils/render';
import CreateTaskPage from '../CreateTaskPage';

vi.mock('../../api/auth', () => ({
  fetchUsers: vi.fn(),
  login: vi.fn(),
  fetchMe: vi.fn(),
}));

vi.mock('../../api/tasks', () => ({
  createTask: vi.fn(),
}));

import { fetchUsers } from '../../api/auth';
import { createTask } from '../../api/tasks';

describe('CreateTaskPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchUsers).mockResolvedValue([
      { id: 1, name: 'Manager One', email: 'm@x.com', role: 'manager' },
      { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
    ]);
  });

  it('shows validation errors when required fields are missing', async () => {
    renderWithRouter(<CreateTaskPage />);
    await waitFor(() => {
      expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /create task/i }));
    expect(await screen.findByText(/title is required/i)).toBeInTheDocument();
    expect(screen.getByText(/please choose an assignee/i)).toBeInTheDocument();
    expect(createTask).not.toHaveBeenCalled();
  });

  it('submits a valid create task form', async () => {
    vi.mocked(createTask).mockResolvedValue({
      id: 99,
      title: 'New task',
      description: '',
      status: 'pending',
      priority: 'medium',
      assigned_to_user_id: 2,
      created_by_user_id: 1,
      assignee: { id: 2, name: 'Anna Analyst', email: 'a@x.com', role: 'analyst' },
      creator: { id: 1, name: 'Manager One', email: 'm@x.com', role: 'manager' },
      created_at: '2026-04-30T00:00:00Z',
      updated_at: '2026-04-30T00:00:00Z',
    });

    renderWithRouter(<CreateTaskPage />);
    await waitFor(() => {
      expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
    });

    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/title/i), 'New task');
    await user.selectOptions(screen.getByLabelText(/assigned to/i), '2');
    await user.click(screen.getByRole('button', { name: /create task/i }));

    await waitFor(() => {
      expect(createTask).toHaveBeenCalledWith({
        title: 'New task',
        description: '',
        priority: 'medium',
        assigned_to_user_id: 2,
      });
    });
  });
});
