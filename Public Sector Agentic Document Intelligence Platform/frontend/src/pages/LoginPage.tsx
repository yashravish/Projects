import { FormEvent, useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { ArrowRight, FileText } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAuth } from '@/state/auth';
import { useToast } from '@/components/ui/Toast';

export function LoginPage() {
  const { login, status } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { push } = useToast();
  const [email, setEmail] = useState('seed-admin@example.gov');
  const [password, setPassword] = useState('ChangeMe!2026');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (status === 'authenticated') {
    const from = (location.state as { from?: string } | null)?.from ?? '/documents';
    return <Navigate to={from} replace />;
  }

  async function handle(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(email, password);
      push('Session opened.', 'success');
      const from = (location.state as { from?: string } | null)?.from ?? '/documents';
      navigate(from, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not sign in';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-[5fr_7fr]">
      {/* Left — masthead, the editorial bit */}
      <section className="relative hidden lg:flex flex-col justify-between border-r border-hair border-rule-soft bg-paper-deep px-12 py-14">
        <header className="stagger">
          <p className="rubric">000 — Masthead</p>
          <h1 className="display text-5xl mt-4 leading-[1.02]">
            The Public<span className="display-italic">Sector</span> <br />
            Agentic Document <br />
            Intelligence Platform.
          </h1>
          <hr className="rule mt-8 w-32" />
          <p className="mt-6 text-base text-ink-80 max-w-prose leading-relaxed">
            An audit-grade reading room for federal, state, and tribal analysts.
            Every retrieved passage is grounded, every answer is cited, every
            decision is recorded — built for the kind of work where being
            <span className="display-italic"> almost right </span>
            is not enough.
          </p>
        </header>

        <div className="grid grid-cols-3 gap-6 stagger">
          <Stat rubric="grounded" value="100%" caption="of cited spans verified" />
          <Stat rubric="latency p95" value="< 4s" caption="from question to answer" />
          <Stat rubric="tenants" value="∞" caption="multi-org by design" />
        </div>

        <footer className="datum text-2xs text-ink-40 uppercase tracking-rubric stagger">
          <p>Filed for internal use — Volume I, Issue 26</p>
        </footer>
      </section>

      {/* Right — the form, kept in restrained whitespace */}
      <section className="flex items-center justify-center px-6 py-10 lg:py-16">
        <div className="w-full max-w-md stagger">
          <div className="mb-9">
            <p className="rubric">002 — Authentication</p>
            <h2 className="display text-4xl mt-2">Sign in.</h2>
            <p className="mt-3 text-sm text-ink-60 leading-relaxed max-w-prose">
              Use your agency credentials. Sessions expire on a fixed schedule;
              refresh tokens rotate on every issuance.
            </p>
          </div>

          <form onSubmit={handle} className="space-y-7" noValidate>
            <Input
              label="Agency email"
              rubric="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              hint={<span>required</span>}
            />
            <Input
              label="Passphrase"
              rubric="passphrase"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              error={error ?? undefined}
            />

            <div className="flex items-center justify-between gap-6 pt-2">
              <Link to="/register" className="btn-ghost">
                <FileText size={14} aria-hidden /> Open a new dossier
              </Link>
              <Button type="submit" loading={submitting} rightIcon={<ArrowRight size={14} />}>
                Continue
              </Button>
            </div>
          </form>

          <hr className="rule-soft mt-12" />

          <p className="mt-5 text-xs text-ink-60 leading-relaxed">
            Seeded administrator on first boot:&nbsp;
            <span className="datum text-ink">seed-admin@example.gov</span> /&nbsp;
            <span className="datum text-ink">ChangeMe!2026</span>.
            Replace these values in <span className="datum">.env</span> before
            promoting to staging.
          </p>
        </div>
      </section>
    </div>
  );
}

function Stat({ rubric, value, caption }: { rubric: string; value: string; caption: string }) {
  return (
    <div>
      <p className="rubric">{rubric}</p>
      <p className="display text-4xl mt-1">{value}</p>
      <p className="datum text-2xs text-ink-60 mt-2 uppercase tracking-[0.06em]">
        {caption}
      </p>
    </div>
  );
}
