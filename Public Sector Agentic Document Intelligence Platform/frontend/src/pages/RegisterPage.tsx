import { FormEvent, useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAuth } from '@/state/auth';
import { useToast } from '@/components/ui/Toast';

export function RegisterPage() {
  const { register, status } = useAuth();
  const navigate = useNavigate();
  const { push } = useToast();
  const [email, setEmail] = useState('');
  const [organizationName, setOrganizationName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (status === 'authenticated') return <Navigate to="/documents" replace />;

  async function handle(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError('Passphrases do not match.');
      return;
    }
    if (password.length < 10) {
      setError('Passphrases must be at least 10 characters.');
      return;
    }
    setSubmitting(true);
    try {
      await register(email, password, organizationName);
      push('Dossier opened. Welcome.', 'success');
      navigate('/documents', { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen grid place-items-center bg-paper px-6 py-12">
      <div className="w-full max-w-2xl panel p-12 stagger">
        <p className="rubric">001 — Open a new dossier</p>
        <h1 className="display text-4xl mt-2">Establish an organization.</h1>
        <p className="mt-3 text-sm text-ink-60 leading-relaxed max-w-prose">
          A new dossier is a fully-isolated tenant. Documents, queries, and
          audit records will be scoped to the organization name you choose
          here. The first user is provisioned as administrator.
        </p>

        <hr className="rule-soft my-8" />

        <form onSubmit={handle} className="grid grid-cols-1 sm:grid-cols-2 gap-x-10 gap-y-7" noValidate>
          <div className="sm:col-span-2">
            <Input
              label="Organization name"
              rubric="organization"
              required
              value={organizationName}
              onChange={(e) => setOrganizationName(e.target.value)}
              placeholder="Office of Resilience Programs"
            />
          </div>
          <Input
            label="Email"
            rubric="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Input
            label="Role"
            rubric="role"
            value="admin (initial)"
            disabled
          />
          <Input
            label="Passphrase"
            rubric="passphrase"
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            hint={<span>min 10 chars</span>}
          />
          <Input
            label="Confirm passphrase"
            rubric="confirm"
            type="password"
            autoComplete="new-password"
            required
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            error={error ?? undefined}
          />

          <div className="sm:col-span-2 flex items-center justify-between gap-6 pt-2">
            <Link to="/login" className="btn-ghost">
              Back to sign-in
            </Link>
            <Button type="submit" loading={submitting} rightIcon={<ArrowRight size={14} />}>
              Establish dossier
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
