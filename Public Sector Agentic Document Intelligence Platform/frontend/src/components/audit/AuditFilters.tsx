import { ChangeEvent } from 'react';
import { Search, X } from 'lucide-react';
import type { AuditEventFilters, AuditOutcome } from '@/api/audit';
import { cn } from '@/lib/cn';

interface Props {
  value: AuditEventFilters;
  onChange: (next: AuditEventFilters) => void;
  /** Distinct values observed in the current page; used to seed multi-selects. */
  knownActions: string[];
  knownResourceTypes: string[];
  knownActors: { id: string; email: string | null }[];
  /** Total events in the current filtered set, for the result-count strap. */
  total: number | null;
}

const OUTCOMES: { id: AuditOutcome; label: string; tone: string }[] = [
  { id: 'success', label: 'success', tone: 'forest' },
  { id: 'denied', label: 'denied', tone: 'leaf' },
  { id: 'error', label: 'error', tone: 'seal' },
];

/**
 * Filters along the top of the timeline. A single horizontal strap with:
 *   - a free-text search (matches action / resource_type / metadata JSON)
 *   - outcome chips (success / denied / error) — toggle pills
 *   - "since" / "until" date inputs
 *   - action prefix select (populated from observed verbs in the page)
 *   - resource type select
 *   - actor select
 *
 * Filter state is fully owned by the parent — this component is a controlled
 * input. Updating any field calls `onChange` with a fresh filter object;
 * the parent decides when to refetch.
 */
export function AuditFilters({
  value,
  onChange,
  knownActions,
  knownResourceTypes,
  knownActors,
  total,
}: Props) {
  const setOutcome = (o: AuditOutcome) => {
    const cur = new Set(value.outcomes ?? []);
    if (cur.has(o)) cur.delete(o);
    else cur.add(o);
    onChange({ ...value, outcomes: cur.size > 0 ? Array.from(cur) : undefined, page: 1 });
  };

  const setSearch = (s: string) =>
    onChange({ ...value, search: s, page: 1 });

  const setSince = (e: ChangeEvent<HTMLInputElement>) =>
    onChange({
      ...value,
      since: e.target.value ? new Date(e.target.value).toISOString() : null,
      page: 1,
    });
  const setUntil = (e: ChangeEvent<HTMLInputElement>) =>
    onChange({
      ...value,
      until: e.target.value ? new Date(e.target.value).toISOString() : null,
      page: 1,
    });

  const setAction = (e: ChangeEvent<HTMLSelectElement>) =>
    onChange({
      ...value,
      actions: e.target.value ? [e.target.value] : undefined,
      page: 1,
    });
  const setResource = (e: ChangeEvent<HTMLSelectElement>) =>
    onChange({
      ...value,
      resource_types: e.target.value ? [e.target.value] : undefined,
      page: 1,
    });
  const setActor = (e: ChangeEvent<HTMLSelectElement>) =>
    onChange({
      ...value,
      actor_ids: e.target.value ? [e.target.value] : undefined,
      page: 1,
    });

  const filtersActive =
    !!value.search ||
    !!value.since ||
    !!value.until ||
    (value.outcomes?.length ?? 0) > 0 ||
    (value.actions?.length ?? 0) > 0 ||
    (value.resource_types?.length ?? 0) > 0 ||
    (value.actor_ids?.length ?? 0) > 0;

  const reset = () => onChange({ page: 1, page_size: value.page_size ?? 50 });

  return (
    <div className="border-y border-hair border-rule-soft py-4 grid gap-4">
      {/* Row 1 — search + outcomes + reset */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <div className="flex items-center gap-2 flex-1 min-w-[16rem]">
          <Search size={14} strokeWidth={1.6} className="text-ink-40" />
          <input
            type="search"
            value={value.search ?? ''}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search action, resource, or metadata…"
            className="field py-1.5 text-sm"
          />
        </div>

        <div className="flex items-center gap-1.5">
          {OUTCOMES.map((o) => {
            const active = (value.outcomes ?? []).includes(o.id);
            return (
              <button
                type="button"
                key={o.id}
                onClick={() => setOutcome(o.id)}
                className={cn(
                  'pill cursor-pointer transition-colors',
                  active && o.tone === 'forest' && 'pill-forest',
                  active && o.tone === 'leaf' && 'pill-leaf',
                  active && o.tone === 'seal' && 'pill-seal',
                )}
                aria-pressed={active}
              >
                {o.label}
              </button>
            );
          })}
        </div>

        {filtersActive ? (
          <button
            type="button"
            onClick={reset}
            className="btn-ghost text-2xs uppercase tracking-rubric"
          >
            <X size={13} strokeWidth={1.6} />
            Clear
          </button>
        ) : null}
      </div>

      {/* Row 2 — typed selects + dates */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-x-6 gap-y-3">
        <Select
          label="action"
          value={value.actions?.[0] ?? ''}
          onChange={setAction}
          options={knownActions}
        />
        <Select
          label="resource"
          value={value.resource_types?.[0] ?? ''}
          onChange={setResource}
          options={knownResourceTypes}
        />
        <SelectActor
          value={value.actor_ids?.[0] ?? ''}
          onChange={setActor}
          options={knownActors}
        />
        <DateField
          label="since"
          value={isoToInputValue(value.since ?? null)}
          onChange={setSince}
        />
        <DateField
          label="until"
          value={isoToInputValue(value.until ?? null)}
          onChange={setUntil}
        />
      </div>

      {total !== null ? (
        <p className="datum text-2xs text-ink-60 uppercase tracking-rubric">
          {filtersActive ? 'filtered set' : 'unfiltered'} — {total.toLocaleString()}{' '}
          {total === 1 ? 'entry' : 'entries'}
        </p>
      ) : null}
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (e: ChangeEvent<HTMLSelectElement>) => void;
  options: string[];
}) {
  return (
    <label className="block">
      <span className="rubric">{label}</span>
      <select
        value={value}
        onChange={onChange}
        className="field py-1.5 text-sm bg-transparent"
      >
        <option value="">all</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function SelectActor({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (e: ChangeEvent<HTMLSelectElement>) => void;
  options: { id: string; email: string | null }[];
}) {
  return (
    <label className="block">
      <span className="rubric">actor</span>
      <select
        value={value}
        onChange={onChange}
        className="field py-1.5 text-sm bg-transparent"
      >
        <option value="">all</option>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.email ?? `${o.id.slice(0, 8)}…`}
          </option>
        ))}
      </select>
    </label>
  );
}

function DateField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className="block">
      <span className="rubric">{label}</span>
      <input
        type="datetime-local"
        value={value}
        onChange={onChange}
        className="field py-1.5 text-sm bg-transparent"
      />
    </label>
  );
}

/**
 * Convert an ISO timestamp into the value `<input type="datetime-local">`
 * expects (`YYYY-MM-DDTHH:mm` in *local* time). Returns the empty string
 * when the input is null.
 */
function isoToInputValue(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
