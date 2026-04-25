import type { LedgerStats } from '@/api/audit';

interface Props {
  stats: LedgerStats;
}

/**
 * Header strip on The Ledger. Tabular numerics, ruled cells, no shadow.
 *
 * Shows five canonical metrics and the chain head / tail hashes. Hashes are
 * truncated to 12 hex chars in the strip — full values appear on the
 * integrity panel and on individual event rows.
 */
export function LedgerStatsStrip({ stats }: Props) {
  return (
    <div className="border-y border-hair border-rule">
      <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
        <Cell
          rubric="i — entries"
          value={fmtInt(stats.total_events)}
          caption={
            stats.last_event_at
              ? `last ${relTime(stats.last_event_at)}`
              : 'no entries yet'
          }
        />
        <Cell
          rubric="ii — past 24 h"
          value={fmtInt(stats.events_24h)}
          caption={pctOfTotal(stats.events_24h, stats.total_events)}
        />
        <Cell
          rubric="iii — past 7 d"
          value={fmtInt(stats.events_7d)}
          caption={pctOfTotal(stats.events_7d, stats.total_events)}
        />
        <Cell
          rubric="iv — actions"
          value={fmtInt(stats.distinct_actions)}
          caption="distinct verbs"
        />
        <Cell
          rubric="v — actors"
          value={fmtInt(stats.distinct_actors)}
          caption="distinct subjects"
        />
      </dl>
      <div className="grid grid-cols-1 md:grid-cols-2 border-t border-hair border-rule-soft">
        <HashCell label="head" value={stats.head_hash} />
        <HashCell label="tail" value={stats.tail_hash} divider />
      </div>
    </div>
  );
}

function Cell({
  rubric,
  value,
  caption,
}: {
  rubric: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="px-5 py-4 border-r border-hair border-rule-soft last:border-r-0">
      <p className="rubric">{rubric}</p>
      <p className="datum text-3xl mt-1.5 leading-none">{value}</p>
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric mt-2">
        {caption}
      </p>
    </div>
  );
}

function HashCell({
  label,
  value,
  divider,
}: {
  label: string;
  value: string | null;
  divider?: boolean;
}) {
  return (
    <div
      className={`px-5 py-3 ${divider ? 'md:border-l border-hair border-rule-soft' : ''}`}
    >
      <p className="rubric mb-1">chain {label}</p>
      <p
        className="datum text-xs text-ink-80 break-all"
        title={value ?? undefined}
      >
        {value ? value : '—'}
      </p>
    </div>
  );
}

function fmtInt(n: number): string {
  return n.toLocaleString();
}

function pctOfTotal(part: number, total: number): string {
  if (total === 0) return '—';
  const pct = (part / total) * 100;
  return `${pct.toFixed(pct >= 10 ? 0 : 1)}% of file`;
}

function relTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const diffSec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`;
  if (diffSec < 86_400) return `${Math.round(diffSec / 3600)}h ago`;
  return `${Math.round(diffSec / 86_400)}d ago`;
}
