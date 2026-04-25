import type { EvalDataset } from '@/api/schemas';
import { cn } from '@/lib/cn';

interface Props {
  dataset: EvalDataset;
}

/** Read-only list of gold dataset items (versioned spec for the harness). */
export function DatasetView({ dataset }: Props) {
  return (
    <div>
      <header className="flex items-baseline justify-between gap-6 mb-4">
        <div>
          <p className="rubric">004.0 — Truth on file</p>
          <h2 className="display text-3xl mt-1">{dataset.name}</h2>
          <p className="text-sm text-ink-80 leading-relaxed max-w-prose mt-2">
            {dataset.description}
          </p>
        </div>
        <div className="text-right shrink-0">
          <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
            version
          </p>
          <p className="datum text-sm text-ink mt-0.5">{dataset.version}</p>
          <p className="datum text-2xs text-ink-40 uppercase tracking-rubric mt-2">
            {dataset.n_items} item{dataset.n_items === 1 ? '' : 's'}
          </p>
        </div>
      </header>

      <ol className="border-t border-hair border-rule">
        {dataset.items.map((it, i) => (
          <li
            key={it.id}
            className={cn(
              'grid grid-cols-[2.75rem_minmax(0,1fr)_minmax(10rem,14rem)_5rem]',
              'items-baseline gap-4 px-1 py-3 border-b border-hair border-rule-soft',
            )}
          >
            <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
              {String(i + 1).padStart(2, '0')}
            </span>
            <span className="display text-base text-ink leading-snug">
              {it.question}
            </span>
            <span className="datum text-2xs text-ink-60 uppercase tracking-rubric truncate">
              {it.expected_doc_filenames[0] ?? '—'}
              {it.expected_doc_filenames.length > 1 && (
                <span className="ml-1 text-ink-40">
                  +{it.expected_doc_filenames.length - 1}
                </span>
              )}
            </span>
            <span className="datum text-2xs text-ink-60 uppercase tracking-rubric text-right">
              {it.topic || '—'}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
