import type { QueryRunListItem } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * Past inquiries roll. A hairline-divided list, with a tiny "PASSED" /
 * "REVIEW" stamp on each row. Rows are clickable; the parent loads the run
 * detail and re-displays it as if it had just run.
 */

interface Props {
  items: QueryRunListItem[];
  activeRunId: string | null;
  onSelect: (runId: string) => void;
}

export function InquiryHistory({ items, activeRunId, onSelect }: Props) {
  if (items.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        No prior inquiries on file.
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
        const passed =
          it.status === 'success' &&
          (it.grounding_score ?? 0) >= 0.7 &&
          (it.hallucination_risk ?? 1) <= 0.3;

        const isActive = activeRunId === it.run_id;
        return (
          <li key={it.run_id}>
            <button
              type="button"
              onClick={() => onSelect(it.run_id)}
              className={cn(
                'w-full text-left grid grid-cols-[7.5rem_1fr_auto] items-baseline gap-5',
                'border-b border-hair border-rule-soft py-4 px-1',
                'transition-colors duration-200',
                isActive
                  ? 'bg-paper-deep/60'
                  : 'hover:bg-paper-deep/30',
              )}
            >
              <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
                {created}
              </span>

              <span className="display text-base text-ink leading-snug truncate">
                {it.question}
              </span>

              <span
                className={cn(
                  'datum text-2xs uppercase tracking-rubric px-2 py-0.5 border-hair',
                  passed
                    ? 'border-forest text-forest bg-forest/5'
                    : it.status === 'failed'
                      ? 'border-seal text-seal bg-seal/5'
                      : 'border-rule-soft text-ink-60',
                )}
              >
                {passed ? 'Passed' : it.status === 'failed' ? 'Failed' : 'Review'}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
