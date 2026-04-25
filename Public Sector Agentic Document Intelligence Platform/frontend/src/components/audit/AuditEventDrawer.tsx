import { useEffect } from 'react';
import { X } from 'lucide-react';
import type { AuditEvent } from '@/api/audit';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/cn';

interface Props {
  event: AuditEvent | null;
  onClose: () => void;
}

/**
 * Right-side detail drawer for one ledger entry. Mounts to a fixed-position
 * overlay so it floats above the page; ESC and click-the-scrim both dismiss.
 *
 * Keeps the dossier aesthetic: hairline frame, no shadow, monotyped hashes,
 * a JSON pretty-print of the metadata. The chain link (prev / entry hashes)
 * is the marquee element of this drawer because that's the cryptographic
 * thread the page is named after.
 */
export function AuditEventDrawer({ event, onClose }: Props) {
  useEffect(() => {
    if (!event) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [event, onClose]);

  if (!event) return null;

  const created = new Date(event.created_at).toLocaleString();
  const outcomeTone =
    event.outcome === 'success'
      ? 'forest'
      : event.outcome === 'denied'
        ? 'leaf'
        : 'seal';

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Audit entry detail"
      className="fixed inset-0 z-40"
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute inset-0 bg-ink/20"
        aria-label="Close"
      />
      <aside
        className={cn(
          'absolute top-0 right-0 h-full w-[min(34rem,90vw)] bg-paper border-l border-hair border-rule',
          'overflow-y-auto animate-rise-in',
        )}
      >
        <div className="px-7 pt-7 pb-4 flex items-baseline justify-between border-b border-hair border-rule-soft">
          <div>
            <p className="rubric">006 — Entry detail</p>
            <h2
              className="display text-2xl mt-1 break-all"
              title={event.event_id}
            >
              {event.action}
            </h2>
            <p className="datum text-2xs text-ink-60 uppercase tracking-rubric mt-1">
              {created}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="btn-ghost"
            aria-label="Close drawer"
          >
            <X size={14} strokeWidth={1.6} />
          </button>
        </div>

        <div className="px-7 py-6 grid gap-6">
          <Section label="i — provenance">
            <Row label="entry id" value={event.event_id} mono />
            <Row label="actor" value={event.actor_email ?? '—'} />
            <Row label="actor id" value={event.actor_id ?? '—'} mono />
            <Row
              label="resource"
              value={`${event.resource_type}${
                event.resource_id ? `/${event.resource_id}` : ''
              }`}
              mono
            />
            <Row label="request id" value={event.request_id ?? '—'} mono />
            <Row
              label="outcome"
              value={<Badge tone={outcomeTone}>{event.outcome}</Badge>}
            />
          </Section>

          <hr className="rule-soft" />

          <Section label="ii — chain link">
            <Row label="prev_hash" value={event.prev_hash ?? '—'} mono wrap />
            <Row label="entry_hash" value={event.entry_hash} mono wrap />
            <p className="text-2xs text-ink-60 mt-1 leading-relaxed">
              entry_hash = SHA-256( canonical_payload || prev_hash ). Re-walk
              the chain on the integrity panel to verify nothing has been
              altered since this entry was written.
            </p>
          </Section>

          <hr className="rule-soft" />

          <Section label="iii — metadata">
            <pre
              className={cn(
                'datum text-2xs text-ink-80 leading-relaxed',
                'border-hair border-rule-soft px-3 py-3 bg-paper-deep/40',
                'whitespace-pre-wrap break-all overflow-x-auto',
              )}
            >
{JSON.stringify(event.metadata, null, 2)}
            </pre>
          </Section>
        </div>
      </aside>
    </div>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="rubric mb-3">{label}</p>
      <div className="grid gap-2.5">{children}</div>
    </div>
  );
}

function Row({
  label,
  value,
  mono,
  wrap,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  wrap?: boolean;
}) {
  return (
    <div className="grid grid-cols-[8rem_1fr] gap-4 items-baseline">
      <dt className="rubric">{label}</dt>
      <dd
        className={cn(
          'text-sm',
          mono && 'datum text-xs',
          wrap ? 'break-all' : 'truncate',
        )}
      >
        {value}
      </dd>
    </div>
  );
}
