import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Download, RefreshCw } from 'lucide-react';
import {
  downloadAuditCsv,
  getLedgerStats,
  listAuditEvents,
  listRetentionPolicies,
  listRetentionRuns,
  runRetention,
  upsertRetentionPolicy,
  verifyIntegrity,
  type AuditEvent,
  type AuditEventFilters,
  type IntegrityReport,
  type RetentionPolicy,
  type RetentionResource,
  type RetentionRun,
} from '@/api/audit';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { useToast } from '@/components/ui/Toast';
import { AuditEventDrawer } from '@/components/audit/AuditEventDrawer';
import { AuditFilters } from '@/components/audit/AuditFilters';
import { AuditTimeline } from '@/components/audit/AuditTimeline';
import { IntegritySeal } from '@/components/audit/IntegritySeal';
import { LedgerStatsStrip } from '@/components/audit/LedgerStatsStrip';
import { RetentionPanel } from '@/components/audit/RetentionPanel';
import { useAuth } from '@/state/auth';

/** Tenant-scoped ledger: stats, integrity check, event stream, retention. Admin-only policy/sweep mutations. */
export function AuditPage() {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';
  const qc = useQueryClient();
  const { push } = useToast();

  const [filters, setFilters] = useState<AuditEventFilters>({
    page: 1,
    page_size: 50,
  });
  const [selected, setSelected] = useState<AuditEvent | null>(null);
  const [exporting, setExporting] = useState(false);

  const statsQuery = useQuery({
    queryKey: ['audit-stats'],
    queryFn: getLedgerStats,
    refetchInterval: 30_000,
  });

  const eventsQuery = useQuery({
    queryKey: ['audit-events', filters],
    queryFn: () => listAuditEvents(filters),
    placeholderData: (prev) => prev,
  });

  const integrityMutation = useMutation<IntegrityReport, Error, void>({
    mutationFn: () => verifyIntegrity(),
    onSuccess: (report) => {
      void qc.invalidateQueries({ queryKey: ['audit-stats'] });
      void qc.invalidateQueries({ queryKey: ['audit-events'] });
      push(
        report.chain_ok
          ? `Chain verified — ${report.total_events.toLocaleString()} entries.`
          : `Chain broken — ${report.breaks.length} ${report.breaks.length === 1 ? 'break' : 'breaks'} detected.`,
        report.chain_ok ? 'success' : 'error',
      );
    },
    onError: (err) => push(err.message, 'error'),
  });

  const integrityQuery = useQuery({
    queryKey: ['audit-integrity-initial'],
    queryFn: () => verifyIntegrity(),
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });
  const integrityReport: IntegrityReport | null =
    integrityMutation.data ?? integrityQuery.data ?? null;

  const policiesQuery = useQuery({
    queryKey: ['retention-policies'],
    queryFn: listRetentionPolicies,
  });
  const runsQuery = useQuery({
    queryKey: ['retention-runs'],
    queryFn: () => listRetentionRuns(1, 25),
  });

  const [pendingResource, setPendingResource] =
    useState<RetentionResource | null>(null);

  const upsertMutation = useMutation<
    RetentionPolicy,
    Error,
    {
      resource: RetentionResource;
      ttl_days: number;
      is_active: boolean;
      notes: string | null;
    }
  >({
    mutationFn: ({ resource, ttl_days, is_active, notes }) =>
      upsertRetentionPolicy(resource, { ttl_days, is_active, notes }),
    onMutate: ({ resource }) => setPendingResource(resource),
    onSuccess: (p) => {
      void qc.invalidateQueries({ queryKey: ['retention-policies'] });
      void qc.invalidateQueries({ queryKey: ['audit-events'] });
      void qc.invalidateQueries({ queryKey: ['audit-stats'] });
      push(
        `Policy saved: ${p.resource_type} → ${p.ttl_days === 0 ? 'forever' : `${p.ttl_days} days`}.`,
        'success',
      );
    },
    onError: (err) => push(err.message, 'error'),
    onSettled: () => setPendingResource(null),
  });

  const sweepMutation = useMutation<RetentionRun, Error, void>({
    mutationFn: () => runRetention(),
    onSuccess: (run) => {
      void qc.invalidateQueries({ queryKey: ['retention-runs'] });
      void qc.invalidateQueries({ queryKey: ['audit-stats'] });
      void qc.invalidateQueries({ queryKey: ['audit-events'] });
      const total = Object.values(run.purged_counts).reduce(
        (a, b) => a + (Number.isFinite(b as number) ? (b as number) : 0),
        0,
      );
      if (run.status === 'failed') {
        push(run.error_message ?? 'Retention sweep failed', 'error');
      } else {
        push(
          total === 0
            ? 'Sweep complete — nothing eligible.'
            : `Sweep complete — ${total} ${total === 1 ? 'row' : 'rows'} purged.`,
          'success',
        );
      }
    },
    onError: (err) => push(err.message, 'error'),
  });

  const knownActions = useMemo(
    () =>
      Array.from(
        new Set(
          (eventsQuery.data?.items ?? []).map((e: AuditEvent) => e.action),
        ),
      ).sort(),
    [eventsQuery.data],
  );
  const knownResourceTypes = useMemo(
    () =>
      Array.from(
        new Set(
          (eventsQuery.data?.items ?? []).map(
            (e: AuditEvent) => e.resource_type,
          ),
        ),
      ).sort(),
    [eventsQuery.data],
  );
  const knownActors = useMemo(() => {
    const items = eventsQuery.data?.items ?? [];
    const seen = new Map<string, string | null>();
    for (const e of items) {
      if (e.actor_id && !seen.has(e.actor_id)) {
        seen.set(e.actor_id, e.actor_email);
      }
    }
    return Array.from(seen.entries()).map(([id, email]) => ({ id, email }));
  }, [eventsQuery.data]);

  useEffect(() => {
    if (sweepMutation.isSuccess) {
      void integrityQuery.refetch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sweepMutation.isSuccess]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const filename = await downloadAuditCsv(filters);
      push(`Exported ${filename}.`, 'success');
      void qc.invalidateQueries({ queryKey: ['audit-stats'] });
      void qc.invalidateQueries({ queryKey: ['audit-events'] });
    } catch (err) {
      push(
        err instanceof Error ? err.message : 'Export failed',
        'error',
      );
    } finally {
      setExporting(false);
    }
  };

  const events = eventsQuery.data?.items ?? [];
  const total = eventsQuery.data?.total ?? null;
  const page = filters.page ?? 1;
  const pageSize = filters.page_size ?? 50;
  const totalPages = total ? Math.max(1, Math.ceil(total / pageSize)) : 1;

  return (
    <div className="stagger">
      <header className="mb-8 grid grid-cols-1 lg:grid-cols-[1fr_auto] items-end gap-6">
        <div>
          <p className="rubric">006 — The Ledger</p>
          <h1 className="display text-5xl mt-2">Read what was done.</h1>
          <p className="mt-4 text-base text-ink-80 max-w-prose leading-relaxed">
            Every action — uploads, queries, evaluations, model promotions,
            sign-ins — is filed here as one immutable entry, cryptographically
            bound to the entry before it. The chain is the chain. Below: the
            entries themselves, the seal that proves them, and the policies
            that decide what data lingers.
          </p>
        </div>
        <div className="flex flex-col items-stretch lg:items-end gap-2">
          <Button
            variant="outline"
            onClick={handleExport}
            loading={exporting}
            leftIcon={<Download size={13} strokeWidth={1.6} />}
          >
            Export CSV
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              void statsQuery.refetch();
              void eventsQuery.refetch();
              void runsQuery.refetch();
              void policiesQuery.refetch();
            }}
            leftIcon={<RefreshCw size={13} strokeWidth={1.6} />}
          >
            Refresh
          </Button>
        </div>
      </header>

      <hr className="rule-double" />

      {/* Stats */}
      <section className="my-8">
        {statsQuery.isLoading ? (
          <Skeleton rows={3} />
        ) : statsQuery.isError ? (
          <ErrorState
            title="Could not load stats"
            description={
              statsQuery.error instanceof Error
                ? statsQuery.error.message
                : 'Unknown error'
            }
            action={
              <Button onClick={() => statsQuery.refetch()}>Retry</Button>
            }
          />
        ) : statsQuery.data ? (
          <LedgerStatsStrip stats={statsQuery.data} />
        ) : null}
      </section>

      {/* Integrity */}
      <section className="my-10">
        <IntegritySeal
          report={integrityReport}
          loading={
            integrityMutation.isPending || integrityQuery.isLoading
          }
          onVerify={() => integrityMutation.mutate()}
        />
      </section>

      <hr className="rule-double" />

      <section className="my-10">
        <header className="flex items-baseline justify-between mb-4">
          <div>
            <p className="rubric">006.2 — Entries on file</p>
            <h2 className="display text-3xl mt-1">Browse the record.</h2>
          </div>
        </header>

        <AuditFilters
          value={filters}
          onChange={setFilters}
          knownActions={knownActions}
          knownResourceTypes={knownResourceTypes}
          knownActors={knownActors}
          total={total}
        />

        <div className="mt-6">
          {eventsQuery.isLoading ? (
            <Skeleton rows={6} />
          ) : eventsQuery.isError ? (
            <ErrorState
              title="Could not load entries"
              description={
                eventsQuery.error instanceof Error
                  ? eventsQuery.error.message
                  : 'Unknown error'
              }
              action={
                <Button onClick={() => eventsQuery.refetch()}>Retry</Button>
              }
            />
          ) : (
            <AuditTimeline
              events={events}
              selectedId={selected?.event_id ?? null}
              onSelect={setSelected}
            />
          )}
        </div>

        {total && total > pageSize ? (
          <nav className="mt-5 flex items-center justify-between datum text-2xs uppercase tracking-rubric text-ink-60">
            <span>
              Page {page} of {totalPages} · {total.toLocaleString()} entries
            </span>
            <div className="flex gap-3">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() =>
                  setFilters((f) => ({ ...f, page: Math.max(1, page - 1) }))
                }
                className="btn-ghost disabled:opacity-30 disabled:cursor-not-allowed"
              >
                ← Previous
              </button>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() =>
                  setFilters((f) => ({
                    ...f,
                    page: Math.min(totalPages, page + 1),
                  }))
                }
                className="btn-ghost disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Next →
              </button>
            </div>
          </nav>
        ) : null}
      </section>

      <hr className="rule-double" />

      {/* Retention */}
      <section className="my-10">
        <header className="mb-5">
          <p className="rubric">006.3 — Retention</p>
          <h2 className="display text-3xl mt-1">
            Decide what lingers, what falls away.
          </h2>
          <p className="text-sm text-ink-80 max-w-prose mt-3 leading-relaxed">
            Per-resource TTLs in days. A sweep enforces every active policy in
            one transaction; soft-deletes are reversible upstream, hard-deletes
            are not. The sweep itself is audited — its row appears in the
            timeline above the moment it completes.
          </p>
        </header>

        {policiesQuery.isLoading || runsQuery.isLoading ? (
          <Skeleton rows={4} />
        ) : policiesQuery.isError ? (
          <ErrorState
            title="Could not load retention policies"
            description={
              policiesQuery.error instanceof Error
                ? policiesQuery.error.message
                : 'Unknown error'
            }
            action={
              <Button onClick={() => policiesQuery.refetch()}>Retry</Button>
            }
          />
        ) : (
          <RetentionPanel
            policies={policiesQuery.data?.items ?? []}
            runs={runsQuery.data?.items ?? []}
            isAdmin={isAdmin}
            busyResource={
              upsertMutation.isPending ? pendingResource : null
            }
            isSweeping={sweepMutation.isPending}
            onSave={(resource, payload) =>
              upsertMutation.mutate({ resource, ...payload })
            }
            onSweep={() => sweepMutation.mutate()}
          />
        )}
      </section>

      <AuditEventDrawer
        event={selected}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
