import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Crosshair, X } from 'lucide-react';
import { predictWithModel } from '@/api/training';
import type {
  RegisteredModelSummary,
  RerankerPredictResponse,
} from '@/api/schemas';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { useToast } from '@/components/ui/Toast';
import { cn } from '@/lib/cn';

interface Props {
  model: RegisteredModelSummary;
  onClose: () => void;
}

const SAMPLE_QUERY = 'When is the grant deadline?';
const SAMPLE_PASSAGES = [
  'Applications must be submitted by February 28, 2026.',
  'Records officers should expect a 12-18% increase in disclosure volume.',
  'The procurement vendor portal is available at portal.gov.',
];

/**
 * Bench-test mini-tool: lets the analyst score (query, passages) against any
 * registered artifact before promoting it. The result is rendered in
 * descending score order with a small bar so the spread is legible.
 */
export function BenchTest({ model, onClose }: Props) {
  const { push } = useToast();
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [passagesText, setPassagesText] = useState(SAMPLE_PASSAGES.join('\n'));

  const mutation = useMutation<RerankerPredictResponse, Error, void>({
    mutationFn: () => {
      const passages = passagesText
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
      if (passages.length === 0) {
        throw new Error('Provide at least one passage to score.');
      }
      if (query.trim().length === 0) {
        throw new Error('Query cannot be blank.');
      }
      return predictWithModel(model.model_id, {
        query: query.trim(),
        passages,
      });
    },
    onError: (err) => {
      push(err.message, 'error');
    },
  });

  const result = mutation.data;
  const max = result?.scored?.length
    ? Math.max(...result.scored.map((s) => s.score))
    : 1;
  const min = result?.scored?.length
    ? Math.min(...result.scored.map((s) => s.score))
    : 0;
  const range = max - min || 1;

  return (
    <section
      className="panel-deep p-7 lg:p-8 relative"
      aria-labelledby="bench-test-title"
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute top-4 right-4 text-ink-60 hover:text-ink p-1"
        aria-label="Close bench test"
      >
        <X size={16} strokeWidth={1.5} />
      </button>

      <header className="mb-6">
        <p className="rubric">005 — bench-test</p>
        <h3 id="bench-test-title" className="display text-3xl mt-1">
          Strike a passage against the model.
        </h3>
        <p className="datum text-2xs text-ink-60 uppercase tracking-rubric mt-2">
          {model.name} · {model.version} · {model.stage}
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-7">
        <div className="space-y-4">
          <div>
            <label
              htmlFor="bench-query"
              className="datum text-2xs text-ink-60 uppercase tracking-rubric"
            >
              Query
            </label>
            <input
              id="bench-query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-1 w-full bg-paper border-b border-hair border-rule px-2 py-2 text-base font-display focus:outline-none focus:border-ink"
              placeholder="A short search-style question…"
            />
          </div>

          <div>
            <label
              htmlFor="bench-passages"
              className="datum text-2xs text-ink-60 uppercase tracking-rubric"
            >
              Passages — one per line
            </label>
            <textarea
              id="bench-passages"
              rows={8}
              value={passagesText}
              onChange={(e) => setPassagesText(e.target.value)}
              className="mt-1 w-full bg-paper border-hair border-rule p-3 text-sm leading-relaxed font-mono focus:outline-none focus:border-ink"
              spellCheck={false}
            />
          </div>

          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            loading={mutation.isPending}
            leftIcon={<Crosshair size={14} strokeWidth={1.6} />}
          >
            Score
          </Button>
        </div>

        <div>
          <p className="rubric">scored — descending</p>
          {mutation.isError ? (
            <ErrorState
              title="Score failed"
              description={mutation.error.message}
            />
          ) : !result ? (
            <p className="text-sm text-ink-60 mt-3 max-w-prose">
              Hit <em>Score</em> to run the (query, passages) pair against the
              model. Output is sorted descending by reranker score with a small
              spread bar so the separation is legible at a glance.
            </p>
          ) : (
            <ul className="mt-3 space-y-3">
              {result.scored.map((row, idx) => {
                const norm = (row.score - min) / range;
                return (
                  <li
                    key={`${row.index}-${idx}`}
                    className="border-hair border-rule-soft pb-2"
                  >
                    <div className="flex items-baseline justify-between gap-4">
                      <span className="datum text-2xs text-ink-40 uppercase tracking-rubric">
                        rank {idx + 1} · #{row.index + 1}
                      </span>
                      <span
                        className={cn(
                          'datum text-base leading-none',
                          idx === 0 ? 'text-seal font-medium' : 'text-ink',
                        )}
                      >
                        {row.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-sm text-ink-80 mt-1 leading-relaxed">
                      {row.passage}
                    </p>
                    <div
                      className="h-px mt-2 bg-ink/15"
                      aria-hidden
                      style={{ width: `${Math.max(norm * 100, 3)}%` }}
                    />
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
