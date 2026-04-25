import { useEffect, useState } from 'react';
import type { TraceStep } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * The agent timeline.
 *
 * Visualises the four-node pipeline (plan → retrieve → synthesize → critique)
 * as a typewriter-style log. While a request is in-flight we render a
 * deterministic *predicted* sequence (so the user sees motion immediately);
 * once the response arrives we replace the predicted entries with the actual
 * trace from the server, preserving timestamps and durations.
 */

interface Props {
  trace: TraceStep[];
  running: boolean;
}

const NODE_INDEX: Record<string, number> = {
  plan: 1,
  retrieve: 2,
  synthesize: 3,
  critique: 4,
  validate: 4,
  error: 99,
};

const NODE_LABEL: Record<string, string> = {
  plan: 'Planning',
  retrieve: 'Retrieving',
  synthesize: 'Synthesizing',
  critique: 'Validating',
  validate: 'Validating',
  error: 'Pipeline failure',
};

const PREDICTED_ORDER: Array<keyof typeof NODE_LABEL> = [
  'plan',
  'retrieve',
  'synthesize',
  'critique',
];

export function AgentTimeline({ trace, running }: Props) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 700);
    return () => window.clearInterval(id);
  }, [running]);

  const rows = running && trace.length === 0
    ? PREDICTED_ORDER.slice(0, Math.min(PREDICTED_ORDER.length, tick + 1)).map(
        (node, i) => ({
          node,
          label: NODE_LABEL[node],
          detail: 'in progress',
          duration_ms: 0,
          predicted: true,
          index: i + 1,
        }),
      )
    : trace.map((s, i) => ({
        node: s.node,
        label: s.label || NODE_LABEL[s.node] || s.node,
        detail: s.detail,
        duration_ms: s.duration_ms,
        predicted: false,
        index: NODE_INDEX[s.node] ?? i + 1,
      }));

  if (rows.length === 0) {
    return (
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        Awaiting inquiry.
      </p>
    );
  }

  return (
    <ol className="border-t border-hair border-rule-soft" aria-live="polite">
      {rows.map((r, i) => (
        <li
          key={`${r.node}-${i}`}
          className={cn(
            'grid grid-cols-[3rem_1fr_auto] items-baseline gap-4 py-3',
            'border-b border-hair border-rule-soft animate-rise-in',
          )}
          style={{ animationDelay: `${i * 70}ms` }}
        >
          <span
            className={cn(
              'datum text-2xs uppercase tracking-rubric',
              r.predicted ? 'text-ink-40 animate-ink-blink' : 'text-seal',
            )}
          >
            {String(r.index).padStart(2, '0')}
          </span>

          <div className="min-w-0">
            <p
              className={cn(
                'display text-base leading-tight',
                r.predicted && 'italic text-ink-60',
              )}
            >
              {r.label}
              {r.predicted && (
                <span className="ml-2 datum text-2xs text-ink-40 lowercase tracking-rubric">
                  in progress…
                </span>
              )}
            </p>
            {!r.predicted && r.detail && (
              <p className="text-xs text-ink-60 mt-1 leading-snug">
                {r.detail}
              </p>
            )}
          </div>

          <span className="datum text-2xs text-ink-40 uppercase tracking-rubric whitespace-nowrap">
            {r.predicted ? '——' : `${r.duration_ms} ms`}
          </span>
        </li>
      ))}
    </ol>
  );
}
