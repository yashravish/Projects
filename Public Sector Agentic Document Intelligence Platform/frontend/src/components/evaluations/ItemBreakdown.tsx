import { useState } from 'react';
import { ChevronRight, FileWarning, BookOpen, AlertTriangle } from 'lucide-react';
import type { EvaluationItem } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * Per-item breakdown row.
 *
 * Each item is a single row in a table-of-record style. Click to expand and
 * read the gold expectations alongside what the agent actually said. The
 * verdict pill on the right is the only colour permitted in the row — every
 * other piece of metadata is an ink/datum line.
 */
interface Props {
  items: EvaluationItem[];
}

export function ItemBreakdown({ items }: Props) {
  if (items.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        No items in scope.
      </p>
    );
  }

  return (
    <ol className="border-t border-hair border-rule">
      {items.map((it, i) => (
        <ItemRow key={it.gold.id} item={it} index={i} />
      ))}
    </ol>
  );
}

function ItemRow({ item, index }: { item: EvaluationItem; index: number }) {
  const [open, setOpen] = useState(false);
  const passed = item.metrics.item_passed;
  const tone = passed ? 'forest' : 'seal';

  return (
    <li
      className={cn(
        'border-b border-hair border-rule-soft animate-rise-in',
      )}
      style={{ animationDelay: `${Math.min(index, 10) * 40}ms` }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'group w-full grid items-baseline gap-4 px-1 py-4 text-left',
          'grid-cols-[2.75rem_minmax(0,1fr)_5.25rem_5.25rem_5.25rem_3.5rem_1rem]',
          'transition-colors duration-200',
          open ? 'bg-paper-deep/60' : 'hover:bg-paper-deep/30',
        )}
        aria-expanded={open}
      >
        <span className="datum text-2xs uppercase tracking-rubric text-ink-60">
          {String(index + 1).padStart(2, '0')}
        </span>

        <span className="display text-base text-ink leading-snug truncate">
          {item.gold.question}
        </span>

        <ScoreCell label="recall" value={item.metrics.retrieval_recall} />
        <ScoreCell label="faith" value={item.metrics.faithfulness} />
        <ScoreCell
          label="ground"
          value={item.metrics.grounding_score}
        />

        <span
          className={cn(
            'datum text-2xs uppercase tracking-rubric px-2 py-0.5 border-hair text-center',
            tone === 'forest'
              ? 'border-forest text-forest bg-forest/5'
              : 'border-seal text-seal bg-seal/5',
          )}
        >
          {passed ? 'Passed' : 'Failed'}
        </span>

        <ChevronRight
          size={14}
          strokeWidth={1.5}
          className={cn(
            'transition-transform duration-200 text-ink-40',
            open && 'rotate-90 text-ink',
          )}
          aria-hidden
        />
      </button>

      {open && <ItemDetail item={item} />}
    </li>
  );
}

function ScoreCell({ label, value }: { label: string; value: number }) {
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

function ItemDetail({ item }: { item: EvaluationItem }) {
  const { gold, inquiry, metrics } = item;
  return (
    <article className="grid lg:grid-cols-[1fr_1.5fr] gap-x-12 gap-y-6 px-6 pb-7 pt-1">
      {/* Left: the expectations. */}
      <section>
        <p className="rubric mb-2">
          <BookOpen size={11} strokeWidth={1.5} className="inline mr-1.5 -mt-px" />
          Truth on file
        </p>
        <dl className="space-y-3 mt-3">
          <Pair
            label="Expected documents"
            value={gold.expected_doc_filenames.join(', ') || '—'}
          />
          <Pair
            label="Required phrases"
            value={
              <ul className="space-y-1 mt-0.5">
                {gold.must_contain_any.map((group, i) => (
                  <li
                    key={i}
                    className="display-italic text-sm text-ink-80 leading-snug"
                  >
                    “{group.join('” / “')}”
                  </li>
                ))}
              </ul>
            }
          />
          {gold.forbidden_phrases.length > 0 && (
            <Pair
              label="Forbidden phrases"
              value={
                <ul className="space-y-1 mt-0.5">
                  {gold.forbidden_phrases.map((p) => (
                    <li
                      key={p}
                      className="display-italic text-sm text-seal/85 leading-snug"
                    >
                      <AlertTriangle
                        size={11}
                        strokeWidth={1.5}
                        className="inline mr-1 -mt-px"
                        aria-hidden
                      />
                      “{p}”
                    </li>
                  ))}
                </ul>
              }
            />
          )}
          <Pair
            label="Topic"
            value={gold.topic || '—'}
          />
        </dl>

        <hr className="rule-soft my-5" />

        <p className="rubric mb-2">Per-item readings</p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-3">
          <Reading label="recall" value={metrics.retrieval_recall} />
          <Reading
            label="precision"
            value={metrics.retrieval_precision}
          />
          <Reading
            label="cite precision"
            value={metrics.citation_precision}
          />
          <Reading label="cite recall" value={metrics.citation_recall} />
          <Reading label="faithfulness" value={metrics.faithfulness} />
          <Reading
            label="forbidden"
            value={metrics.forbidden_phrase_rate}
            invert
          />
          <Reading label="grounding" value={metrics.grounding_score} />
          <Reading
            label="halluc risk"
            value={metrics.hallucination_risk}
            invert
          />
          <Reading
            label="latency"
            value={metrics.latency_ms}
            format="ms"
          />
          <Reading
            label="critic"
            text={metrics.answer_passed_critic ? 'pass' : 'fail'}
            tone={metrics.answer_passed_critic ? 'forest' : 'seal'}
          />
        </div>
      </section>

      {/* Right: the agent's testimony. */}
      <section>
        <p className="rubric mb-2">
          <FileWarning size={11} strokeWidth={1.5} className="inline mr-1.5 -mt-px" />
          The agent's testimony
        </p>
        {inquiry.error ? (
          <p className="display text-base text-seal leading-snug">
            {inquiry.error}
          </p>
        ) : (
          <blockquote className="display text-base leading-relaxed text-ink whitespace-pre-wrap border-l border-rule-soft pl-4">
            {inquiry.answer_text || '(empty)'}
          </blockquote>
        )}

        {inquiry.citations.length > 0 && (
          <>
            <hr className="rule-soft my-5" />
            <p className="rubric mb-2">Cited evidence</p>
            <ul className="space-y-3">
              {inquiry.citations.map((c, i) => (
                <li
                  key={i}
                  className="grid grid-cols-[3.25rem_1fr] items-start gap-3"
                >
                  <span className="datum text-2xs text-ink-60 uppercase tracking-rubric border-hair border-rule-soft px-1.5 py-1 text-center">
                    EX&nbsp;{String(i + 1).padStart(2, '0')}
                  </span>
                  <div className="min-w-0">
                    <p
                      className="datum text-2xs text-ink uppercase tracking-rubric truncate"
                      title={c.document_filename}
                    >
                      {c.document_filename}
                      <span className="text-ink-40 normal-case">
                        {' '}— page {c.page_start}
                        {c.page_end !== c.page_start && `–${c.page_end}`}
                      </span>
                    </p>
                    <p className="display-italic text-sm text-ink-80 leading-snug mt-1">
                      {trim(c.snippet, 220)}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </article>
  );
}

function Pair({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="datum text-2xs uppercase tracking-rubric text-ink-60">
        {label}
      </dt>
      <dd className="text-sm text-ink-80 leading-snug mt-1">{value}</dd>
    </div>
  );
}

interface ReadingProps {
  label: string;
  value?: number;
  text?: string;
  invert?: boolean;
  format?: 'percent' | 'ms';
  tone?: 'forest' | 'seal' | 'neutral';
}

function Reading({
  label,
  value,
  text,
  invert,
  format = 'percent',
  tone,
}: ReadingProps) {
  let display = text;
  let resolvedTone: 'forest' | 'seal' | 'neutral' = tone ?? 'neutral';
  if (display === undefined && value !== undefined) {
    if (format === 'ms') {
      display = `${Math.round(value)} ms`;
    } else {
      display = `${(value * 100).toFixed(0)}%`;
      const passes = invert ? value <= 0.3 : value >= 0.7;
      resolvedTone = passes ? 'forest' : 'seal';
    }
  }
  return (
    <div>
      <p className="datum text-2xs uppercase tracking-rubric text-ink-40">
        {label}
      </p>
      <p
        className={cn(
          'datum text-base leading-none mt-0.5',
          resolvedTone === 'forest' && 'text-forest',
          resolvedTone === 'seal' && 'text-seal',
        )}
      >
        {display ?? '—'}
      </p>
    </div>
  );
}

function trim(s: string, max: number): string {
  const flat = s.replace(/\s+/g, ' ').trim();
  if (flat.length <= max) return flat;
  return flat.slice(0, max - 1).replace(/[.,;:\s]+$/, '') + '…';
}
