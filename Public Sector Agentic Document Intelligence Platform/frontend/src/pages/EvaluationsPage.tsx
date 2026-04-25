import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Beaker, RefreshCw, FileSearch } from 'lucide-react';
import {
  getDataset,
  getEvaluationRun,
  listEvaluationRuns,
  postEvaluationRun,
} from '@/api/evaluations';
import type { EvaluationRunDetail } from '@/api/schemas';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { EmptyState } from '@/components/ui/EmptyState';
import { useToast } from '@/components/ui/Toast';
import { DatasetView } from '@/components/evaluations/DatasetView';
import { ItemBreakdown } from '@/components/evaluations/ItemBreakdown';
import { MetricsStrip } from '@/components/evaluations/MetricsStrip';
import { RunHistory } from '@/components/evaluations/RunHistory';

/** Run and inspect gold-dataset evaluations: dataset spec, run detail, per-item breakdown, history. */
export function EvaluationsPage() {
  const qc = useQueryClient();
  const { push } = useToast();

  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvaluationRunDetail | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);

  const dataset = useQuery({
    queryKey: ['eval-dataset'],
    queryFn: getDataset,
  });

  const history = useQuery({
    queryKey: ['eval-runs'],
    queryFn: () => listEvaluationRuns(1, 25),
  });

  // Auto-select the most recent run on first load if none is selected.
  useEffect(() => {
    if (
      !activeRunId &&
      history.data &&
      history.data.items.length > 0 &&
      !detail
    ) {
      const first = history.data.items[0];
      setActiveRunId(first.run_id);
      detailQuery.mutate(first.run_id);
    }
    // We deliberately depend only on the loaded history — we don't want this
    // to fire again as the user clicks around.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history.data]);

  const detailQuery = useMutation({
    mutationFn: (runId: string) => getEvaluationRun(runId),
    onSuccess: (resp) => {
      setDetail(resp);
      setActiveRunId(resp.run_id);
      setTimeout(
        () =>
          detailRef.current?.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          }),
        50,
      );
    },
    onError: (err: unknown) => {
      push(err instanceof Error ? err.message : 'Could not load run', 'error');
    },
  });

  const runMutation = useMutation({
    mutationFn: () => postEvaluationRun({}),
    onMutate: () => {
      setDetail(null);
      setActiveRunId(null);
    },
    onSuccess: (resp) => {
      setDetail(resp);
      setActiveRunId(resp.run_id);
      void qc.invalidateQueries({ queryKey: ['eval-runs'] });
      const passRate = resp.aggregate.pass_rate;
      if (resp.status === 'failed') {
        push(`Evaluation failed`, 'error');
      } else if (passRate >= 0.7) {
        push(
          `Evaluation passed — ${(passRate * 100).toFixed(0)}% of items.`,
          'success',
        );
      } else {
        push(
          `Evaluation completed at ${(passRate * 100).toFixed(0)}% pass rate; review failures.`,
          'info',
        );
      }
      setTimeout(
        () =>
          detailRef.current?.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          }),
        50,
      );
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : 'Evaluation failed';
      push(msg, 'error');
    },
  });

  const running = runMutation.isPending;

  return (
    <div className="stagger">
      {/* Header. */}
      <header className="mb-8 grid grid-cols-1 lg:grid-cols-[1fr_auto] items-end gap-6">
        <div>
          <p className="rubric">004 — The Audit Bench</p>
          <h1 className="display text-5xl mt-2">Inspect the agent.</h1>
          <p className="mt-4 text-base text-ink-80 max-w-prose leading-relaxed">
            The harness drives every gold-question case through the live
            inquiry pipeline, scores retrieval, citation, and answer
            faithfulness, and persists each run for comparison. Numbers below
            are read directly off the persisted record &mdash; no re-running.
          </p>
        </div>

        <div className="flex flex-col items-stretch lg:items-end gap-2">
          <Button
            onClick={() => runMutation.mutate()}
            disabled={running}
            loading={running}
            leftIcon={<Beaker size={14} strokeWidth={1.6} />}
          >
            {running ? 'Running…' : 'Run the harness'}
          </Button>
          {dataset.data && (
            <p className="datum text-2xs text-ink-40 uppercase tracking-rubric text-right">
              {dataset.data.n_items} items · v{dataset.data.version.slice(0, 8)}
            </p>
          )}
        </div>
      </header>

      <hr className="rule-double" />

      {/* Detail. */}
      <section ref={detailRef} className="my-10">
        <header className="flex items-baseline justify-between mb-5">
          <div>
            <p className="rubric">004.1 — Aggregate readings</p>
            <h2 className="display text-3xl mt-1">
              {detail
                ? `Run ${detail.run_id.slice(0, 8)}`
                : running
                  ? 'In flight…'
                  : 'No run selected.'}
            </h2>
            {detail && (
              <p className="datum text-2xs text-ink-60 uppercase tracking-rubric mt-2">
                {new Date(detail.created_at).toLocaleString()} ·{' '}
                {detail.dataset_name} v{detail.dataset_version.slice(0, 8)} ·{' '}
                model {detail.model} ·{' '}
                {detail.wall_time_ms.toLocaleString()} ms wall
                {detail.mlflow_run_id && (
                  <>
                    {' '}
                    · mlflow {detail.mlflow_run_id.slice(0, 8)}
                  </>
                )}
              </p>
            )}
          </div>
        </header>

        {running ? (
          <div className="border-y border-hair border-rule py-10 text-center">
            <p className="datum text-2xs uppercase tracking-rubric text-ink animate-ink-blink">
              Driving every gold-question case through the live agent…
            </p>
            <p className="text-sm text-ink-60 mt-2 max-w-prose mx-auto">
              On the offline embedder this typically completes in under 30
              seconds. The page will scroll to the readings when finished.
            </p>
          </div>
        ) : runMutation.isError ? (
          <ErrorState
            title="Evaluation failed"
            description={
              runMutation.error instanceof Error
                ? runMutation.error.message
                : 'Unknown error'
            }
            action={
              <Button onClick={() => runMutation.mutate()}>Retry</Button>
            }
          />
        ) : detailQuery.isPending ? (
          <Skeleton rows={4} />
        ) : detail ? (
          <>
            <MetricsStrip aggregate={detail.aggregate} />

            <header className="mt-12 mb-3">
              <p className="rubric">004.2 — Per-item breakdown</p>
              <h3 className="display text-2xl mt-1">
                {detail.aggregate.n_items - detail.aggregate.n_failures} of{' '}
                {detail.aggregate.n_items} cleared.
              </h3>
            </header>
            <ItemBreakdown items={detail.items} />
          </>
        ) : (
          <EmptyState
            rubric="awaiting selection"
            title="No run loaded yet."
            description="Trigger a fresh run with the button above, or pick one from the roll below to inspect it."
          />
        )}
      </section>

      <hr className="rule-double" />

      {/* History. */}
      <section className="my-10">
        <header className="flex items-baseline justify-between mb-5">
          <div>
            <p className="rubric">004.3 — The roll</p>
            <h2 className="display text-3xl mt-1">Past audits.</h2>
          </div>
          <Button
            variant="ghost"
            onClick={() => void history.refetch()}
            leftIcon={<RefreshCw size={13} />}
          >
            Refresh
          </Button>
        </header>

        {history.isLoading ? (
          <Skeleton rows={3} />
        ) : history.isError ? (
          <ErrorState
            title="Could not load past runs"
            description={
              history.error instanceof Error
                ? history.error.message
                : 'Unknown error'
            }
            action={<Button onClick={() => history.refetch()}>Retry</Button>}
          />
        ) : (
          <RunHistory
            items={history.data?.items ?? []}
            activeRunId={activeRunId}
            onSelect={(id) => detailQuery.mutate(id)}
          />
        )}
      </section>

      <hr className="rule-double" />

      {/* Dataset register. */}
      <section className="my-10">
        <header className="flex items-baseline justify-between mb-5">
          <div className="flex items-baseline gap-2">
            <FileSearch
              size={14}
              strokeWidth={1.5}
              className="text-ink-60 -mb-px"
              aria-hidden
            />
            <p className="rubric">Always visible — the spec the agent is judged against.</p>
          </div>
        </header>
        {dataset.isLoading ? (
          <Skeleton rows={6} />
        ) : dataset.isError ? (
          <ErrorState
            title="Could not load dataset"
            description={
              dataset.error instanceof Error
                ? dataset.error.message
                : 'Unknown error'
            }
            action={<Button onClick={() => dataset.refetch()}>Retry</Button>}
          />
        ) : dataset.data ? (
          <DatasetView dataset={dataset.data} />
        ) : null}
      </section>
    </div>
  );
}
