import { type FormEvent, useState } from 'react';
import { Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { ApiError } from '../api/client';
import ErrorMessage from '../components/ErrorMessage';

export default function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const from = (location.state as { from?: string } | null)?.from ?? '/dashboard';
  const sessionExpired = searchParams.get('reason') === 'session';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [touched, setTouched] = useState(false);

  if (user) {
    return <Navigate to={from} replace />;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched(true);
    setError(null);
    if (!email || !password) {
      setError('Please enter your email and password.');
      return;
    }
    setSubmitting(true);
    try {
      await login(email, password);
      navigate(from, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Unable to sign in. Please try again.');
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <section className="auth-card" aria-labelledby="login-heading">
        <h1 id="login-heading">Sign in</h1>
        <p className="lede">
          Workflow Desk — task intake, approvals, and audit-ready reporting for your team.
        </p>
        <form onSubmit={handleSubmit} noValidate className="form-grid">
          {sessionExpired && (
            <div className="alert info" role="status">
              Your session expired. Sign in again to continue.
            </div>
          )}
          <ErrorMessage message={error} />
          <div>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              aria-invalid={touched && !email}
            />
            {touched && !email && (
              <div className="field-error">Email is required.</div>
            )}
          </div>
          <div>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              aria-invalid={touched && !password}
            />
            {touched && !password && (
              <div className="field-error">Password is required.</div>
            )}
          </div>
          <button type="submit" className="primary" disabled={submitting}>
            {submitting ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
        <div className="demo-creds" aria-label="Demo credentials">
          <strong>Demo credentials</strong>
          Manager: manager@example.com / password123
          <br />
          Analyst: analyst@example.com / password123
        </div>
      </section>
    </div>
  );
}
