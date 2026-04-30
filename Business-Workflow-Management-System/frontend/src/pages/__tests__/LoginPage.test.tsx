import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderWithRouter } from '../../test-utils/render';
import LoginPage from '../LoginPage';
import { AuthProvider } from '../../auth/AuthContext';
import { ApiError } from '../../api/client';

vi.mock('../../api/auth', () => ({
  login: vi.fn(),
  fetchMe: vi.fn().mockRejectedValue(new Error('not logged in')),
  fetchUsers: vi.fn(),
}));

import { login } from '../../api/auth';

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  function renderLogin() {
    return renderWithRouter(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
      { initialPath: '/login' }
    );
  }

  it('renders the login form with email and password fields', async () => {
    renderLogin();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
    });
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows demo credentials hint', async () => {
    renderLogin();
    await waitFor(() => {
      expect(screen.getByText(/manager@example.com/)).toBeInTheDocument();
    });
    expect(screen.getByText(/analyst@example.com/)).toBeInTheDocument();
  });

  it('shows an error when login fails', async () => {
    vi.mocked(login).mockRejectedValueOnce(
      new ApiError('Invalid email or password.', 401)
    );
    renderLogin();
    await waitFor(() => {
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    });
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/email/i), 'nope@test.com');
    await user.type(screen.getByLabelText(/password/i), 'wrong');
    await user.click(screen.getByRole('button', { name: /sign in/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      /invalid email or password/i
    );
  });

  it('shows validation error when fields are empty', async () => {
    renderLogin();
    await waitFor(() => {
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /sign in/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      /please enter your email and password/i
    );
  });
});
