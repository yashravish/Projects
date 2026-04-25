import type { Critique } from '@/api/schemas';
import { cn } from '@/lib/cn';

/**
 * The big "PASSED" / "REQUIRES REVIEW" stamp that goes near the answer.
 *
 * Visual: a slightly rotated rectangle with a small-caps verdict, the two
 * scores, and (if any) the issues list. Echoes the seal-stamp aesthetic of
 * a physical dossier without falling into pastiche — no actual rotated PNG
 * stamps, just typography.
 */

interface Props {
  critique: Critique;
  className?: string;
}

export function CritiqueStamp({ critique, className }: Props) {
  const passed = critique.passed;
  const tone = passed ? 'forest' : 'seal';
  return (
    <aside
      className={cn(
        'inline-flex flex-col gap-1.5 px-4 py-3 border-hair',
        tone === 'forest'
          ? 'border-forest text-forest'
          : 'border-seal text-seal',
        className,
      )}
      aria-label={passed ? 'Validation passed' : 'Validation requires review'}
    >
      <p className="datum text-2xs uppercase tracking-rubric">
        {passed ? 'Validated' : 'Requires review'}
      </p>
      <div className="flex items-baseline gap-4">
        <ScoreCell label="grounding" value={critique.grounding_score} />
        <ScoreCell label="hallucination risk" value={critique.hallucination_risk} />
      </div>
      {critique.issues.length > 0 && (
        <ul className="mt-1 text-xs text-ink-80 list-disc pl-4 max-w-md">
          {critique.issues.slice(0, 4).map((issue, i) => (
            <li key={i} className="leading-snug">
              {issue}
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}

function ScoreCell({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <p className="datum text-2xs text-ink-60 uppercase tracking-[0.06em]">
        {label}
      </p>
      <p className="display text-2xl datum leading-none">{value.toFixed(2)}</p>
    </div>
  );
}
