import { ArrowUpCircle, Archive, FlaskConical } from 'lucide-react';
import type { RegisteredModelSummary } from '@/api/schemas';
import { cn } from '@/lib/cn';

interface Props {
  items: RegisteredModelSummary[];
  busyId: string | null;
  onPromote: (modelId: string) => void;
  onArchive: (modelId: string) => void;
  onTest: (modelId: string) => void;
}

const STAGE_LABEL: Record<RegisteredModelSummary['stage'], string> = {
  staging: 'staging',
  production: 'in service',
  archived: 'archived',
};

const STAGE_TONE: Record<RegisteredModelSummary['stage'], string> = {
  staging: 'border-leaf text-leaf bg-leaf/5',
  production: 'border-seal text-seal bg-seal/5',
  archived: 'border-rule text-ink-60 bg-paper-deep/40',
};

export function ModelRegistry({
  items,
  busyId,
  onPromote,
  onArchive,
  onTest,
}: Props) {
  if (items.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        Registry empty &mdash; the forge has not been struck.
      </p>
    );
  }

  return (
    <ul className="border-t border-hair border-rule">
      {items.map((it) => {
        const created = new Date(it.created_at).toLocaleString(undefined, {
          month: 'short',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
        });
        const stage = it.stage;
        const busy = busyId === it.model_id;
        return (
          <li
            key={it.model_id}
            className={cn(
              'grid items-baseline gap-5 px-1 py-4 border-b border-hair border-rule-soft',
              'grid-cols-[7rem_minmax(0,1fr)_4.5rem_4.5rem_4.5rem_6rem_auto]',
              busy && 'opacity-60',
            )}
          >
            <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
              {created}
            </span>

            <span className="display text-base text-ink leading-snug truncate">
              {it.name}
              <span className="datum text-2xs text-ink-40 ml-2 normal-case">
                {it.version}
              </span>
            </span>

            <Stat label="F1" value={it.holdout_f1} pct />
            <Stat label="AUC" value={it.holdout_roc_auc} pct />
            <Stat label="sep" value={it.score_separation} digits={2} />

            <span
              className={cn(
                'datum text-2xs uppercase tracking-rubric px-2 py-0.5 border-hair text-center',
                STAGE_TONE[stage],
              )}
            >
              {STAGE_LABEL[stage]}
            </span>

            <span className="flex items-center gap-1 justify-end">
              <RowAction
                title="Bench-test"
                onClick={() => onTest(it.model_id)}
                disabled={busy}
                icon={<FlaskConical size={14} strokeWidth={1.5} />}
              />
              {stage !== 'production' && (
                <RowAction
                  title="Promote to production"
                  onClick={() => onPromote(it.model_id)}
                  disabled={busy}
                  icon={<ArrowUpCircle size={14} strokeWidth={1.5} />}
                />
              )}
              {stage !== 'archived' && (
                <RowAction
                  title="Archive"
                  onClick={() => onArchive(it.model_id)}
                  disabled={busy}
                  icon={<Archive size={14} strokeWidth={1.5} />}
                />
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Stat({
  label,
  value,
  pct,
  digits = 0,
}: {
  label: string;
  value: number;
  pct?: boolean;
  digits?: number;
}) {
  return (
    <div className="text-right">
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        {label}
      </p>
      <p className="datum text-base text-ink leading-none mt-0.5">
        {pct
          ? (value * 100).toFixed(digits === 0 ? 0 : digits)
          : value.toFixed(digits === 0 ? 2 : digits)}
        {pct && <span className="text-2xs text-ink-40">%</span>}
      </p>
    </div>
  );
}

function RowAction({
  title,
  onClick,
  disabled,
  icon,
}: {
  title: string;
  onClick: () => void;
  disabled?: boolean;
  icon: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center w-8 h-8',
        'border-hair border-rule-soft text-ink-60',
        'transition-colors hover:text-ink hover:bg-paper-deep/50',
        'disabled:opacity-40 disabled:hover:bg-transparent',
      )}
    >
      {icon}
    </button>
  );
}
