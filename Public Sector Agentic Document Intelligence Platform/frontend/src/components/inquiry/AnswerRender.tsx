import { Fragment, useMemo } from 'react';
import type { Citation } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * Renders an answer string with `[N]` markers replaced by interactive
 * citation chips. Clicking a chip notifies the page so the matching evidence
 * card on the right rail can be focused / highlighted.
 *
 * Markers that do not bind to a known citation are rendered as a faint
 * superscript so the reader can see the agent attempted to cite something
 * out-of-range — the critic will have flagged this in `critique.issues`.
 */

interface Props {
  text: string;
  citations: Citation[];
  highlightedIndex: number | null;
  onCitationClick: (index: number) => void;
}

interface Tokenisation {
  segments: Array<{ kind: 'text'; value: string } | { kind: 'cite'; index: number }>;
}

const MARKER = /\[(\d{1,2})\]/g;

function tokenise(text: string): Tokenisation {
  const segments: Tokenisation['segments'] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(MARKER)) {
    const start = match.index ?? 0;
    if (start > lastIndex) {
      segments.push({ kind: 'text', value: text.slice(lastIndex, start) });
    }
    segments.push({ kind: 'cite', index: Number(match[1]) });
    lastIndex = start + match[0].length;
  }
  if (lastIndex < text.length) {
    segments.push({ kind: 'text', value: text.slice(lastIndex) });
  }
  return { segments };
}

export function AnswerRender({
  text,
  citations,
  highlightedIndex,
  onCitationClick,
}: Props) {
  const tokens = useMemo(() => tokenise(text), [text]);
  const knownIndices = useMemo(
    () => new Set(citations.map((c) => c.index)),
    [citations],
  );

  return (
    <div className="display text-lg leading-[1.85] text-ink whitespace-pre-wrap">
      {tokens.segments.map((seg, i) => {
        if (seg.kind === 'text') {
          return <Fragment key={i}>{seg.value}</Fragment>;
        }
        const known = knownIndices.has(seg.index);
        const highlighted = highlightedIndex === seg.index;
        if (!known) {
          return (
            <sup
              key={i}
              className="datum text-2xs text-ink-40 line-through ml-0.5"
              title="Citation marker did not bind to a retrieved chunk."
            >
              [{seg.index}]
            </sup>
          );
        }
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCitationClick(seg.index)}
            className={cn(
              'mx-0.5 align-baseline inline-flex items-center justify-center',
              'datum text-[0.6875rem] uppercase tracking-rubric',
              'px-1.5 py-0.5 border-hair transition-colors duration-200',
              highlighted
                ? 'border-seal text-seal bg-seal/5'
                : 'border-rule-soft text-ink-80 hover:border-seal hover:text-seal',
            )}
            aria-label={`Show evidence ${seg.index}`}
            data-citation-marker={seg.index}
          >
            {String(seg.index).padStart(2, '0')}
          </button>
        );
      })}
    </div>
  );
}
