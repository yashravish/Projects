import type { EvaluationRunSummary } from '@/api/schemas';
import { cn } from '@/lib/cn';

interface Props {
  items: EvaluationRunSummary[];
  activeRunId: string | null;
  onSelect: (runId: string) => void;
}

/**
 * The roll of past evaluation runs. One row per run, click to load detail.
 */
export function RunHistory({ items, activeRunId, onSelect }: Props) {
  if (items.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        No prior evaluation runs on file.
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
        const tone =
          it.status === 'failed'
            ? 'seal'
            : it.pass_rate >= 0.7
              ? 'forest'
              : 'seal';
        const isActive = activeRunId === it.run_id;
        return (
          <li key={it.run_id}>
            <button
              type="button"
              onClick={() => onSelect(it.run_id)}
              className={cn(
                'w-full text-left grid items-baseline gap-5 px-1 py-4',
                'grid-cols-[7rem_minmax(0,1fr)_5rem_5rem_5rem_5rem]',
                'border-b border-hair border-rule-soft transition-colors duration-200',
                isActive ? 'bg-paper-deep/60' : 'hover:bg-paper-deep/30',
              )}
              aria-current={isActive ? 'true' : undefined}
            >
              <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
                {created}
              </span>

              <span className="display text-base text-ink leading-snug truncate">
                {it.dataset_name}
                <span className="datum text-2xs text-ink-40 ml-2 normal-case">
                  v{it.dataset_version.slice(0, 8)}
                </span>
              </span>

              <Stat label="pass" value={it.pass_rate} />
              <Stat label="recall" value={it.retrieval_recall} />
              <Stat label="ground" value={it.grounding_score} />

              <span
                className={cn(
                  'datum text-2xs uppercase tracking-rubric px-2 py-0.5 border-hair text-center',
                  tone === 'forest'
                    ? 'border-forest text-forest bg-forest/5'
                    : 'border-seal text-seal bg-seal/5',
                )}
              >
                {it.status === 'failed'
                  ? 'failed'
                  : it.pass_rate >= 0.7
                    ? 'green'
                    : 'review'}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-right">
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        {label}
      </p>
      <p className="datum text-base text-ink leading-none mt-0.5">
        {(value * 100).toFixed(0)}
        <span className="text-2xs text-ink-40">%</span>
      </p>
    </div>
  );
}
