import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Anvil, RefreshCw, Hammer } from 'lucide-react';
import {
  listRegisteredModels,
  listTrainingJobs,
  promoteModel,
  submitTrainingJob,
} from '@/api/training';
import type {
  RegisteredModelSummary,
  TrainingJobDetail,
} from '@/api/schemas';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { useToast } from '@/components/ui/Toast';
import { ProductionModelCard } from '@/components/models/ProductionModelCard';
import { ModelRegistry } from '@/components/models/ModelRegistry';
import { TrainingJobsRoll } from '@/components/models/TrainingJobsRoll';
import { BenchTest } from '@/components/models/BenchTest';

/** Model registry UI: production slot, bench test, version table, training job history. */
export function ModelsPage() {
  const qc = useQueryClient();
  const { push } = useToast();

  const [autoPromote, setAutoPromote] = useState(false);
  const [benchModelId, setBenchModelId] = useState<string | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);

  const modelsQuery = useQuery({
    queryKey: ['registered-models'],
    queryFn: () => listRegisteredModels(1, 50),
  });

  const jobsQuery = useQuery({
    queryKey: ['training-jobs'],
    queryFn: () => listTrainingJobs(1, 25),
    refetchInterval: 4_000,
    refetchIntervalInBackground: false,
  });

  const trainMutation = useMutation<TrainingJobDetail, Error, void>({
    mutationFn: () =>
      submitTrainingJob({
        name: 'psdi-cross-encoder-reranker',
        auto_promote: autoPromote,
        notes: null,
      }),
    onSuccess: (resp) => {
      void qc.invalidateQueries({ queryKey: ['registered-models'] });
      void qc.invalidateQueries({ queryKey: ['training-jobs'] });
      if (resp.status === 'success') {
        push(
          autoPromote
            ? `Forged ${resp.version.slice(0, 14)}… and put into service.`
            : `Forged ${resp.version.slice(0, 14)}… (staging).`,
          'success',
        );
      } else {
        push(
          resp.error_message ?? 'Training failed; see the roll for details.',
          'error',
        );
      }
    },
    onError: (err) => {
      push(err.message, 'error');
    },
  });

  const promoteMutation = useMutation<
    RegisteredModelSummary,
    Error,
    { modelId: string; stage: 'production' | 'archived' }
  >({
    mutationFn: async ({ modelId, stage }) => {
      const detail = await promoteModel(modelId, { stage });
      return {
        model_id: detail.model_id,
        name: detail.name,
        version: detail.version,
        framework: detail.framework,
        backend: detail.backend,
        stage: detail.stage,
        holdout_f1: detail.metrics.holdout_f1 ?? 0,
        holdout_roc_auc: detail.metrics.holdout_roc_auc ?? 0,
        score_separation: detail.metrics.score_separation ?? 0,
        n_train: detail.metrics.n_train ?? 0,
        artifact_uri: detail.artifact_uri,
        training_job_id: detail.training_job_id,
        created_at: detail.created_at,
        promoted_at: detail.promoted_at ?? null,
        archived_at: detail.archived_at ?? null,
      };
    },
    onMutate: ({ modelId }) => {
      setPendingId(modelId);
    },
    onSuccess: (row) => {
      void qc.invalidateQueries({ queryKey: ['registered-models'] });
      push(
        row.stage === 'production'
          ? `${row.name} ${row.version.slice(0, 12)}… is in service.`
          : `${row.name} ${row.version.slice(0, 12)}… archived.`,
        row.stage === 'production' ? 'success' : 'info',
      );
    },
    onError: (err) => {
      push(err.message, 'error');
    },
    onSettled: () => {
      setPendingId(null);
    },
  });

  const items = modelsQuery.data?.items ?? [];
  const productionModel = useMemo(
    () => items.find((m) => m.stage === 'production') ?? null,
    [items],
  );
  const benchModel = useMemo(
    () => items.find((m) => m.model_id === benchModelId) ?? null,
    [items, benchModelId],
  );

  // If the user clicks bench on a row but the row vanishes (e.g. archived in
  // another tab), close the panel.
  useEffect(() => {
    if (benchModelId && !benchModel) setBenchModelId(null);
  }, [benchModelId, benchModel]);

  const training = trainMutation.isPending;

  return (
    <div className="stagger">
      {/* Header */}
      <header className="mb-8 grid grid-cols-1 lg:grid-cols-[1fr_auto] items-end gap-6">
        <div>
          <p className="rubric">005 — The Forge</p>
          <h1 className="display text-5xl mt-2">Strike, season, and serve.</h1>
          <p className="mt-4 text-base text-ink-80 max-w-prose leading-relaxed">
            Train a cross-encoder reranker on the seeded corpus and the gold
            register, register the artifact, and promote it into service. Once
            promoted, every inquiry passes its top retrieval candidates through
            this model before the answer agent reads them.
          </p>
        </div>

        <div className="flex flex-col items-stretch lg:items-end gap-3">
          <Button
            onClick={() => trainMutation.mutate()}
            disabled={training}
            loading={training}
            leftIcon={<Hammer size={14} strokeWidth={1.6} />}
          >
            {training ? 'At the anvil…' : 'Strike a model'}
          </Button>
          <label className="flex items-center gap-2 text-2xs uppercase tracking-rubric text-ink-60 datum cursor-pointer">
            <input
              type="checkbox"
              checked={autoPromote}
              onChange={(e) => setAutoPromote(e.target.checked)}
              className="accent-seal"
            />
            promote on success
          </label>
        </div>
      </header>

      <hr className="rule-double" />

      {/* In service */}
      <section className="my-10">
        <header className="mb-5">
          <p className="rubric">005.1 — In service</p>
          <h2 className="display text-3xl mt-1">
            {productionModel
              ? 'The model on the bench.'
              : 'Nothing on the bench.'}
          </h2>
        </header>

        {modelsQuery.isLoading ? (
          <Skeleton rows={4} />
        ) : modelsQuery.isError ? (
          <ErrorState
            title="Could not load registry"
            description={
              modelsQuery.error instanceof Error
                ? modelsQuery.error.message
                : 'Unknown error'
            }
            action={
              <Button onClick={() => modelsQuery.refetch()}>Retry</Button>
            }
          />
        ) : (
          <ProductionModelCard
            model={productionModel}
            onTest={(id) => setBenchModelId(id)}
            onArchive={(id) =>
              promoteMutation.mutate({ modelId: id, stage: 'archived' })
            }
            busy={pendingId === productionModel?.model_id}
          />
        )}

        {benchModel && (
          <div className="mt-7">
            <BenchTest
              model={benchModel}
              onClose={() => setBenchModelId(null)}
            />
          </div>
        )}
      </section>

      <hr className="rule-double" />

      {/* Registry */}
      <section className="my-10">
        <header className="flex items-baseline justify-between mb-5">
          <div>
            <p className="rubric">005.2 — The registry</p>
            <h2 className="display text-3xl mt-1">Every version, on file.</h2>
          </div>
          <Button
            variant="ghost"
            onClick={() => void modelsQuery.refetch()}
            leftIcon={<RefreshCw size={13} />}
          >
            Refresh
          </Button>
        </header>

        {modelsQuery.isLoading ? (
          <Skeleton rows={3} />
        ) : modelsQuery.isError ? (
          <ErrorState
            title="Could not load registry"
            description={
              modelsQuery.error instanceof Error
                ? modelsQuery.error.message
                : 'Unknown error'
            }
            action={
              <Button onClick={() => modelsQuery.refetch()}>Retry</Button>
            }
          />
        ) : (
          <ModelRegistry
            items={items}
            busyId={pendingId}
            onPromote={(id) =>
              promoteMutation.mutate({ modelId: id, stage: 'production' })
            }
            onArchive={(id) =>
              promoteMutation.mutate({ modelId: id, stage: 'archived' })
            }
            onTest={(id) => setBenchModelId(id)}
          />
        )}
      </section>

      <hr className="rule-double" />

      {/* Jobs */}
      <section className="my-10">
        <header className="flex items-baseline justify-between mb-5">
          <div className="flex items-baseline gap-2">
            <Anvil
              size={14}
              strokeWidth={1.5}
              className="text-ink-60 -mb-px"
              aria-hidden
            />
            <p className="rubric">005.3 — The roll</p>
          </div>
          <Button
            variant="ghost"
            onClick={() => void jobsQuery.refetch()}
            leftIcon={<RefreshCw size={13} />}
          >
            Refresh
          </Button>
        </header>
        <h2 className="display text-3xl mt-1 mb-5">Strikes, fair and foul.</h2>

        {jobsQuery.isLoading ? (
          <Skeleton rows={3} />
        ) : jobsQuery.isError ? (
          <ErrorState
            title="Could not load training jobs"
            description={
              jobsQuery.error instanceof Error
                ? jobsQuery.error.message
                : 'Unknown error'
            }
            action={<Button onClick={() => jobsQuery.refetch()}>Retry</Button>}
          />
        ) : (
          <TrainingJobsRoll items={jobsQuery.data?.items ?? []} />
        )}
      </section>
    </div>
  );
}
