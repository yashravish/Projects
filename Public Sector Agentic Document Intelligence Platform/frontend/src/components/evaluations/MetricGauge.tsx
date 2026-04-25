import { cn } from '@/lib/cn';

/**
 * A "marginalia gauge" — a hairline meter for a single 0..1 score.
 *
 * The bar is one ink-line tall. The fill draws a second ink-line on top
 * of it for the proportion of the value, then a tick at the threshold.
 * We deliberately do NOT animate the fill: this is an instrument panel,
 * not a dashboard. Numbers move when the data does.
 */
interface Props {
  label: string;
  value: number;
  /** If provided, drawn as a tick on the bar. */
  threshold?: number;
  /** True if smaller is better (e.g. hallucination_risk). Inverts colour. */
  inverted?: boolean;
  className?: string;
  format?: 'percent' | 'fraction' | 'ms';
  /** Sub-rubric printed below the value (e.g. "across 10 items"). */
  hint?: string;
}

export function MetricGauge({
  label,
  value,
  threshold,
  inverted = false,
  className,
  format = 'percent',
  hint,
}: Props) {
  const clamped = Math.max(0, Math.min(1, value));
  const pct = format === 'ms' ? Math.min(1, value / 5000) : clamped;
  const isGood = inverted ? clamped <= (threshold ?? 0.3) : clamped >= (threshold ?? 0.7);
  const tone = isGood ? 'forest' : 'seal';

  const display =
    format === 'percent'
      ? `${(clamped * 100).toFixed(0)}%`
      : format === 'ms'
        ? `${Math.round(value).toLocaleString()} ms`
        : value.toFixed(2);

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <p className="datum text-2xs uppercase tracking-rubric text-ink-60">
        {label}
      </p>

      <div className="flex items-baseline gap-3">
        <span
          className={cn(
            'display datum text-3xl leading-none',
            tone === 'seal' ? 'text-seal' : 'text-ink',
          )}
        >
          {display}
        </span>
        {hint && (
          <span className="datum text-2xs uppercase tracking-rubric text-ink-40">
            {hint}
          </span>
        )}
      </div>

      <div className="relative h-px w-full bg-ink-20" aria-hidden>
        <div
          className={cn(
            'absolute left-0 top-0 h-px',
            tone === 'forest' ? 'bg-forest' : 'bg-seal',
          )}
          style={{ width: `${(pct * 100).toFixed(2)}%` }}
        />
        {threshold !== undefined && format !== 'ms' && (
          <span
            className="absolute -top-[3px] h-[7px] w-px bg-ink-60"
            style={{
              left: `${(threshold * 100).toFixed(2)}%`,
            }}
          />
        )}
      </div>
    </div>
  );
}
