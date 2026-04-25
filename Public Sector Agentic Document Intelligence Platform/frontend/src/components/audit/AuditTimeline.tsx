import { useMemo } from 'react';
import type { AuditEvent, AuditOutcome } from '@/api/audit';
import { cn } from '@/lib/cn';

interface Props {
  events: AuditEvent[];
  selectedId: string | null;
  onSelect: (e: AuditEvent) => void;
}

/**
 * Vertical chronological feed of ledger events, grouped by date.
 *
 * Each row is dense, monotyped, and highly scannable:
 *
 *   HH:MM:SS  action.verb           resource_type/abcdef…   actor@org.gov   ✓
 *
 * Selected rows pin a 0.5px ink-coloured strap on the left; hovering rows
 * underlines the action. The ordering inside a day group is descending
 * (most-recent first) — the same order the API returns. Day headers
 * separate the groups with a labelled hairline rule.
 *
 * Outcome glyph at the row tail:
 *   success → forest dot
 *   denied  → leaf dot
 *   error   → seal dot
 */
export function AuditTimeline({ events, selectedId, onSelect }: Props) {
  const grouped = useMemo(() => groupByDay(events), [events]);

  if (events.length === 0) {
    return (
      <div className="border-y border-hair border-rule-soft py-12 text-center">
        <p className="rubric mb-2">empty leaf</p>
        <h3 className="display text-2xl">No entries match.</h3>
        <p className="text-sm text-ink-60 mt-2">
          Adjust the filters above, or wait — every action in the platform
          will be filed here.
        </p>
      </div>
    );
  }

  return (
    <div className="border-t border-hair border-rule">
      {grouped.map((day) => (
        <section key={day.label}>
          <header className="bg-paper-deep/40 px-1 py-2 border-b border-hair border-rule-soft flex items-baseline justify-between">
            <p className="rubric">{day.label}</p>
            <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
              {day.items.length} {day.items.length === 1 ? 'entry' : 'entries'}
            </p>
          </header>
          <ul>
            {day.items.map((ev) => (
              <Row
                key={ev.event_id}
                event={ev}
                active={selectedId === ev.event_id}
                onClick={() => onSelect(ev)}
              />
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

function Row({
  event,
  active,
  onClick,
}: {
  event: AuditEvent;
  active: boolean;
  onClick: () => void;
}) {
  const time = new Date(event.created_at).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          'w-full grid grid-cols-[5rem_minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_3.5rem_0.5rem]',
          'gap-4 items-baseline px-1 py-2.5 text-left',
          'border-b border-hair border-rule-soft transition-colors',
          'hover:bg-paper-deep/40',
          active && 'bg-paper-deep/60',
        )}
        aria-current={active ? 'true' : undefined}
      >
        <span
          className={cn(
            'datum text-2xs uppercase tracking-rubric',
            active ? 'text-ink' : 'text-ink-60',
          )}
        >
          {time}
        </span>

        <span
          className={cn(
            'datum text-sm tabular-nums truncate',
            event.outcome === 'denied'
              ? 'text-leaf-deep'
              : event.outcome === 'error'
                ? 'text-seal'
                : 'text-ink',
          )}
        >
          {event.action}
        </span>

        <span className="datum text-2xs text-ink-60 truncate">
          {event.resource_type}
          {event.resource_id ? (
            <span className="text-ink-40">/{event.resource_id.slice(0, 8)}</span>
          ) : null}
        </span>

        <span className="text-2xs text-ink-60 truncate">
          {event.actor_email ??
            (event.actor_id ? `${event.actor_id.slice(0, 8)}…` : 'system')}
        </span>

        <span className="text-right">
          <OutcomeGlyph outcome={event.outcome} />
        </span>

        {/* Selection rule */}
        <span
          className={cn(
            'block h-full w-px',
            active ? 'bg-ink' : 'bg-transparent',
          )}
          aria-hidden
        />
      </button>
    </li>
  );
}

function OutcomeGlyph({ outcome }: { outcome: AuditOutcome }) {
  const tone =
    outcome === 'success'
      ? 'bg-forest text-paper'
      : outcome === 'denied'
        ? 'bg-leaf-deep text-paper'
        : 'bg-seal text-paper';
  const ch = outcome === 'success' ? '✓' : outcome === 'denied' ? '∅' : '!';
  return (
    <span
      className={cn(
        'inline-flex items-center justify-center w-4 h-4 rounded-full datum text-2xs',
        tone,
      )}
      title={outcome}
      aria-label={outcome}
    >
      {ch}
    </span>
  );
}

interface DayGroup {
  label: string;
  items: AuditEvent[];
}

function groupByDay(events: AuditEvent[]): DayGroup[] {
  const out: DayGroup[] = [];
  let current: DayGroup | null = null;
  for (const e of events) {
    const d = new Date(e.created_at);
    const label = d.toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: '2-digit',
      year: 'numeric',
    });
    if (!current || current.label !== label) {
      current = { label, items: [] };
      out.push(current);
    }
    current.items.push(e);
  }
  return out;
}
