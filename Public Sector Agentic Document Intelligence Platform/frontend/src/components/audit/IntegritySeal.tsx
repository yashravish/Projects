import { CheckCircle2, AlertTriangle, RefreshCcw } from 'lucide-react';
import type { IntegrityReport } from '@/api/audit';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/cn';

interface Props {
  report: IntegrityReport | null;
  loading: boolean;
  onVerify: () => void;
}

/**
 * The seal panel — a tamper-evidence statement at a glance.
 *
 * Two faces:
 *   verified  → forest "VERIFIED" seal, head/tail hashes printed in full.
 *   broken    → seal turns to seal-red, the offending rows are listed.
 *
 * The verify button can be re-clicked to re-walk the chain at any time.
 */
export function IntegritySeal({ report, loading, onVerify }: Props) {
  const ok = report?.chain_ok ?? null;
  const verifiedAt = report?.verified_at
    ? new Date(report.verified_at).toLocaleString()
    : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[18rem_1fr] gap-0 border-y border-hair border-rule">
      {/* Seal face */}
      <div
        className={cn(
          'p-7 border-r border-hair border-rule-soft flex flex-col items-start gap-4',
          ok === false ? 'bg-seal/5' : ok === true ? 'bg-forest/5' : '',
        )}
      >
        <div
          className={cn(
            'w-20 h-20 rounded-full border-hair flex items-center justify-center',
            ok === false
              ? 'border-seal text-seal'
              : ok === true
                ? 'border-forest text-forest'
                : 'border-rule-soft text-ink-40',
          )}
          aria-hidden
        >
          {ok === false ? (
            <AlertTriangle size={28} strokeWidth={1.5} />
          ) : (
            <CheckCircle2
              size={32}
              strokeWidth={1.5}
              className={loading ? 'animate-ink-blink' : ''}
            />
          )}
        </div>
        <div>
          <p className="rubric">tamper-evidence</p>
          <p
            className={cn(
              'display text-3xl mt-1 leading-none',
              ok === false
                ? 'text-seal'
                : ok === true
                  ? 'text-forest'
                  : 'text-ink-60',
            )}
          >
            {report
              ? ok
                ? 'Verified.'
                : 'Broken.'
              : loading
                ? 'Verifying…'
                : 'Unverified.'}
          </p>
          {verifiedAt ? (
            <p className="datum text-2xs text-ink-60 uppercase tracking-rubric mt-3">
              walked {verifiedAt}
            </p>
          ) : null}
        </div>
        <Button
          variant="outline"
          onClick={onVerify}
          loading={loading}
          leftIcon={<RefreshCcw size={13} strokeWidth={1.6} />}
        >
          Re-walk the chain
        </Button>
      </div>

      {/* Body */}
      <div className="p-7">
        <p className="rubric">006.1 — Chain integrity</p>
        <h3 className="display text-2xl mt-1">
          {report
            ? ok
              ? `${report.total_events.toLocaleString()} entries, all in order.`
              : `${report.breaks.length.toLocaleString()} ${report.breaks.length === 1 ? 'break' : 'breaks'} detected in ${report.total_events.toLocaleString()} entries.`
            : 'Press the seal to verify.'}
        </h3>
        <p className="text-sm text-ink-80 max-w-prose mt-3 leading-relaxed">
          Every audit entry carries a SHA-256 fingerprint of its payload bound
          to the entry before it, anchored to the tenant. Re-walking the chain
          re-derives every fingerprint server-side; the moment two adjacent
          entries disagree, this panel turns red and the offending row(s) are
          named below.
        </p>

        {report ? (
          <dl className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-y-2 gap-x-6">
            <Field label="head" value={report.head_hash} />
            <Field label="tail" value={report.tail_hash} />
          </dl>
        ) : null}

        {report && !ok && report.breaks.length > 0 ? (
          <div className="mt-6 border-hair border-rule">
            <p className="rubric px-3 py-2 border-b border-hair border-rule-soft text-seal">
              breaks
            </p>
            <ul>
              {report.breaks.slice(0, 10).map((b) => (
                <li
                  key={b.event_id}
                  className="px-3 py-3 text-xs datum border-b border-hair border-rule-soft last:border-b-0"
                >
                  <p className="text-ink-80">
                    event {b.event_id.slice(0, 8)} ·{' '}
                    {new Date(b.created_at).toLocaleString()}
                  </p>
                  <p className="text-ink-40 mt-1 break-all">
                    expected {short(b.expected_entry_hash)} · observed{' '}
                    <span className="text-seal">
                      {short(b.observed_entry_hash)}
                    </span>
                  </p>
                </li>
              ))}
            </ul>
            {report.breaks.length > 10 ? (
              <p className="px-3 py-2 datum text-2xs text-ink-40 uppercase tracking-rubric">
                + {report.breaks.length - 10} more
              </p>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div>
      <dt className="rubric">{label}</dt>
      <dd className="datum text-xs text-ink-80 mt-0.5 break-all">
        {value ?? '—'}
      </dd>
    </div>
  );
}

function short(h: string): string {
  if (h.length <= 16) return h;
  return `${h.slice(0, 8)}…${h.slice(-6)}`;
}
