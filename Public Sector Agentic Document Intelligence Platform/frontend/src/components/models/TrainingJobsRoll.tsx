import type { TrainingJobSummary } from '@/api/schemas';
import { cn } from '@/lib/cn';

interface Props {
  items: TrainingJobSummary[];
}

const STATUS_TONE: Record<TrainingJobSummary['status'], string> = {
  success: 'border-forest text-forest bg-forest/5',
  failed: 'border-seal text-seal bg-seal/5',
  running: 'border-leaf text-leaf bg-leaf/5 animate-ink-blink',
  pending: 'border-rule text-ink-60 bg-paper-deep/40',
};

export function TrainingJobsRoll({ items }: Props) {
  if (items.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        No training jobs on file.
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
        return (
          <li
            key={it.job_id}
            className={cn(
              'grid items-baseline gap-5 px-1 py-4 border-b border-hair border-rule-soft',
              'grid-cols-[7rem_minmax(0,1fr)_5rem_5rem_5rem_5rem_5rem]',
            )}
          >
            <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
              {created}
            </span>
            <span className="display text-base text-ink leading-snug truncate">
              {it.name}
              <span className="datum text-2xs text-ink-40 ml-2 normal-case">
                {it.version} · {it.backend}
              </span>
              {it.error_message && (
                <span className="block text-2xs text-seal mt-1 normal-case truncate font-normal">
                  {it.error_message}
                </span>
              )}
            </span>
            <Stat label="F1" value={it.holdout_f1} pct />
            <Stat label="AUC" value={it.holdout_roc_auc} pct />
            <Stat
              label="sep"
              value={it.score_separation}
              digits={2}
            />
            <Stat
              label="dur"
              value={it.duration_s}
              digits={1}
              suffix="s"
            />
            <span
              className={cn(
                'datum text-2xs uppercase tracking-rubric px-2 py-0.5 border-hair text-center',
                STATUS_TONE[it.status],
              )}
            >
              {it.status}
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
  suffix,
}: {
  label: string;
  value: number;
  pct?: boolean;
  digits?: number;
  suffix?: string;
}) {
  return (
    <div className="text-right">
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        {label}
      </p>
      <p className="datum text-base text-ink leading-none mt-0.5">
        {pct
          ? (value * 100).toFixed(digits === 0 ? 0 : digits)
          : value.toFixed(digits)}
        {pct && <span className="text-2xs text-ink-40">%</span>}
        {suffix && (
          <span className="text-2xs text-ink-40 ml-0.5">{suffix}</span>
        )}
      </p>
    </div>
  );
}
