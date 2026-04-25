import { forwardRef } from 'react';
import { ExternalLink } from 'lucide-react';
import type { Citation } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * One evidence card on the right rail.
 *
 * Visual: an EXHIBIT stamp number, a SMALL CAPS mono filename, a folio
 * ("page N–M"), and an italic Newsreader snippet. When the marker in the
 * answer is hovered or focused the card receives `highlighted=true` and
 * gets an ink-bleed underline.
 */

interface Props {
  citation: Citation;
  highlighted?: boolean;
  onClick?: () => void;
}

export const CitationCard = forwardRef<HTMLElement, Props>(function CitationCard(
  { citation: c, highlighted, onClick },
  ref,
) {
  return (
    <article
      ref={ref}
      onClick={onClick}
      className={cn(
        'group relative grid grid-cols-[3.25rem_1fr] items-start gap-4 py-5',
        'border-b border-hair border-rule-soft transition-colors duration-200',
        highlighted ? 'bg-paper-deep/60' : 'hover:bg-paper-deep/30',
        onClick && 'cursor-pointer',
      )}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if (!onClick) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      data-citation-index={c.index}
    >
      {/* The stamp. */}
      <div
        className={cn(
          'flex items-center justify-center border-hair px-2 py-1.5',
          'datum text-2xs uppercase tracking-rubric',
          highlighted
            ? 'border-seal text-seal bg-seal/5'
            : 'border-rule-soft text-ink-60',
        )}
        aria-hidden
      >
        EX&nbsp;{String(c.index).padStart(2, '0')}
      </div>

      <div className="min-w-0">
        <header className="flex items-center justify-between gap-3">
          <p
            className="datum text-2xs uppercase tracking-rubric text-ink truncate"
            title={c.document_filename}
          >
            {c.document_filename}
          </p>
          <ExternalLink
            size={12}
            strokeWidth={1.5}
            className={cn(
              'flex-none transition-opacity',
              highlighted ? 'opacity-90 text-seal' : 'opacity-30 group-hover:opacity-70',
            )}
            aria-hidden
          />
        </header>

        <p className="datum text-2xs text-ink-60 mt-1 uppercase tracking-rubric">
          page {c.page_start}{c.page_end !== c.page_start && `–${c.page_end}`}
        </p>

        <blockquote
          className={cn(
            'mt-3 display-italic text-[0.95rem] leading-snug text-ink-80',
            'border-l-hair border-rule-soft pl-3',
          )}
        >
          {trimSnippet(c.snippet)}
        </blockquote>
      </div>
    </article>
  );
});

function trimSnippet(s: string, max = 280): string {
  const flat = s.replace(/\s+/g, ' ').trim();
  if (flat.length <= max) return flat;
  return flat.slice(0, max - 1).replace(/[.,;:\s]+$/, '') + '…';
}
