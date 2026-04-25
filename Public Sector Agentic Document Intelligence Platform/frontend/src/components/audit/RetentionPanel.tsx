import { useMemo, useState } from 'react';
import { Play, Save, ShieldOff } from 'lucide-react';
import type {
  RetentionPolicy,
  RetentionResource,
  RetentionRun,
} from '@/api/audit';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/cn';

interface Props {
  policies: RetentionPolicy[];
  runs: RetentionRun[];
  isAdmin: boolean;
  busyResource: RetentionResource | null;
  isSweeping: boolean;
  onSave: (
    resource: RetentionResource,
    payload: { ttl_days: number; is_active: boolean; notes: string | null },
  ) => void;
  onSweep: () => void;
}

const RESOURCES: { id: RetentionResource; label: string; description: string }[] = [
  {
    id: 'document',
    label: 'Documents',
    description:
      'Soft-deletes uploaded source documents and their chunks. Audit rows mentioning the document remain.',
  },
  {
    id: 'query_run',
    label: 'Query runs',
    description:
      'Hard-deletes user inquiries and the trace persisted alongside them. Aggregate counts in the ledger are preserved.',
  },
  {
    id: 'evaluation_run',
    label: 'Evaluation runs',
    description:
      'Hard-deletes harness runs (per-item answers + metrics). Useful when the dataset version is the only thing you need to keep.',
  },
];

/** Per-resource retention TTLs, sweep action, and sweep history. Audit log retention is platform-owned. */
export function RetentionPanel({
  policies,
  runs,
  isAdmin,
  busyResource,
  isSweeping,
  onSave,
  onSweep,
}: Props) {
  const byResource = useMemo(() => {
    const m = new Map<RetentionResource, RetentionPolicy>();
    for (const p of policies) m.set(p.resource_type, p);
    return m;
  }, [policies]);

  const lastRun = runs[0] ?? null;

  return (
    <div>
      <div className="border-y border-hair border-rule">
        <div className="grid grid-cols-[10rem_8rem_minmax(0,1fr)_8rem] gap-5 px-1 py-3 border-b border-hair border-rule-soft bg-paper-deep/40">
          <p className="rubric">resource</p>
          <p className="rubric">ttl (days)</p>
          <p className="rubric">notes</p>
          <p className="rubric text-right">action</p>
        </div>
        {RESOURCES.map((r) => (
          <PolicyRow
            key={r.id}
            resource={r}
            existing={byResource.get(r.id) ?? null}
            isAdmin={isAdmin}
            busy={busyResource === r.id}
            onSave={(payload) => onSave(r.id, payload)}
          />
        ))}
        <ImmunityRow />
      </div>

      <div
        className={cn(
          'mt-6 border-hair border-rule grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-x-6 items-center px-5 py-4',
          lastRun?.status === 'failed' ? 'bg-seal/5' : 'bg-paper-deep/40',
        )}
      >
        <div>
          <p className="rubric">last sweep</p>
          {lastRun ? (
            <p className="display text-xl mt-1">
              {lastRun.status === 'success'
                ? `${sumPurged(lastRun)} ${sumPurged(lastRun) === 1 ? 'row' : 'rows'} purged`
                : lastRun.status === 'failed'
                  ? 'Failed'
                  : 'In flight…'}
              <span className="datum text-2xs text-ink-60 ml-3 uppercase tracking-rubric">
                {new Date(lastRun.started_at).toLocaleString()}
              </span>
            </p>
          ) : (
            <p className="display text-xl mt-1 text-ink-60">No sweeps yet.</p>
          )}
          {lastRun?.error_message ? (
            <p className="text-xs text-seal mt-2">{lastRun.error_message}</p>
          ) : null}
          {lastRun?.status === 'success' ? (
            <p className="datum text-2xs text-ink-60 mt-2 uppercase tracking-rubric">
              {Object.entries(lastRun.purged_counts)
                .filter(([, n]) => (n as number) > 0)
                .map(([k, n]) => `${k}: ${n}`)
                .join(' · ') || 'nothing eligible'}
            </p>
          ) : null}
        </div>

        <Button
          onClick={onSweep}
          disabled={!isAdmin || isSweeping}
          loading={isSweeping}
          leftIcon={<Play size={13} strokeWidth={1.6} />}
        >
          {isAdmin ? 'Run sweep now' : 'Admin only'}
        </Button>
      </div>

      {runs.length > 1 ? (
        <div className="mt-5">
          <p className="rubric mb-2">earlier sweeps</p>
          <ul className="border-t border-hair border-rule-soft">
            {runs.slice(1, 11).map((r) => (
              <li
                key={r.run_id}
                className="grid grid-cols-[8rem_minmax(0,1fr)_5rem] items-baseline gap-5 px-1 py-2 border-b border-hair border-rule-soft"
              >
                <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
                  {new Date(r.started_at).toLocaleString()}
                </span>
                <span className="datum text-xs text-ink-80 truncate">
                  {r.status === 'success'
                    ? Object.entries(r.purged_counts)
                        .map(([k, n]) => `${k}:${n}`)
                        .join(' · ') || '0 rows eligible'
                    : r.error_message ?? 'pending'}
                </span>
                <span className="text-right">
                  <Badge
                    tone={
                      r.status === 'success'
                        ? 'forest'
                        : r.status === 'failed'
                          ? 'seal'
                          : 'leaf'
                    }
                  >
                    {r.status}
                  </Badge>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function PolicyRow({
  resource,
  existing,
  isAdmin,
  busy,
  onSave,
}: {
  resource: { id: RetentionResource; label: string; description: string };
  existing: RetentionPolicy | null;
  isAdmin: boolean;
  busy: boolean;
  onSave: (p: {
    ttl_days: number;
    is_active: boolean;
    notes: string | null;
  }) => void;
}) {
  const [ttl, setTtl] = useState<string>(
    existing ? String(existing.ttl_days) : '',
  );
  const [active, setActive] = useState<boolean>(existing?.is_active ?? true);
  const [notes, setNotes] = useState<string>(existing?.notes ?? '');

  /* If the existing policy refreshes from the server, re-sync the local form
     unless the user has been editing. We compare against the controlled
     values to detect a "dirty" form. */
  const baselineTtl = existing ? String(existing.ttl_days) : '';
  const baselineActive = existing?.is_active ?? true;
  const baselineNotes = existing?.notes ?? '';
  const dirty =
    ttl !== baselineTtl || active !== baselineActive || notes !== baselineNotes;

  const ttlInt = Number.parseInt(ttl, 10);
  const ttlValid = Number.isFinite(ttlInt) && ttlInt >= 0 && ttlInt <= 36500;

  const handleSave = () => {
    if (!ttlValid) return;
    onSave({
      ttl_days: ttlInt,
      is_active: active,
      notes: notes.trim().length === 0 ? null : notes.trim(),
    });
  };

  const updatedAt = existing
    ? new Date(existing.updated_at).toLocaleDateString(undefined, {
        month: 'short',
        day: '2-digit',
        year: 'numeric',
      })
    : null;

  return (
    <div className="grid grid-cols-[10rem_8rem_minmax(0,1fr)_8rem] gap-5 px-1 py-4 border-b border-hair border-rule-soft items-center">
      <div>
        <p
          className={cn(
            'display text-base leading-tight',
            !active && existing ? 'text-ink-40' : '',
          )}
        >
          {resource.label}
        </p>
        <p className="datum text-2xs text-ink-40 uppercase tracking-rubric mt-1">
          {existing
            ? active
              ? `last edited ${updatedAt}`
              : 'paused'
            : 'no policy yet'}
        </p>
      </div>

      <div>
        <input
          type="number"
          min={0}
          max={36_500}
          value={ttl}
          onChange={(e) => setTtl(e.target.value)}
          placeholder="0 = forever"
          disabled={!isAdmin || busy}
          className="field py-1.5 datum text-base"
        />
      </div>

      <div className="grid gap-1">
        <input
          type="text"
          value={notes}
          maxLength={2000}
          onChange={(e) => setNotes(e.target.value)}
          placeholder={resource.description}
          disabled={!isAdmin || busy}
          className="field py-1.5 text-sm"
        />
        {isAdmin ? (
          <label className="flex items-center gap-2 text-2xs uppercase tracking-rubric text-ink-60 datum cursor-pointer">
            <input
              type="checkbox"
              checked={active}
              onChange={(e) => setActive(e.target.checked)}
              disabled={busy}
              className="accent-seal"
            />
            active
          </label>
        ) : null}
      </div>

      <div className="text-right">
        {isAdmin ? (
          <Button
            variant={dirty ? 'primary' : 'outline'}
            onClick={handleSave}
            disabled={!dirty || !ttlValid || busy}
            loading={busy}
            leftIcon={<Save size={13} strokeWidth={1.6} />}
          >
            Save
          </Button>
        ) : (
          <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
            read-only
          </p>
        )}
      </div>
    </div>
  );
}

function ImmunityRow() {
  return (
    <div className="grid grid-cols-[10rem_minmax(0,1fr)] gap-5 px-1 py-3 items-center bg-paper-deep/30">
      <div className="flex items-center gap-2">
        <ShieldOff
          size={14}
          strokeWidth={1.6}
          className="text-ink-60"
          aria-hidden
        />
        <p className="datum text-xs uppercase tracking-rubric text-ink-60">
          audit_log
        </p>
      </div>
      <p className="text-xs text-ink-60 italic">
        Immune to retention. The chain is the chain — the platform never
        deletes ledger entries.
      </p>
    </div>
  );
}

function sumPurged(run: RetentionRun): number {
  return Object.values(run.purged_counts).reduce(
    (a, b) => a + (Number.isFinite(b as number) ? (b as number) : 0),
    0,
  );
}
