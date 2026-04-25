import { Anvil, ShieldCheck, Activity, Clock } from 'lucide-react';
import type { RegisteredModelSummary } from '@/api/schemas';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/cn';

interface Props {
  model: RegisteredModelSummary | null;
  onArchive?: (modelId: string) => void;
  onTest?: (modelId: string) => void;
  busy?: boolean;
}

/**
 * The "in service" model. Big, declarative, one-of-a-kind on the page.
 * Distinct from the registry table: this is the answer to "what is currently
 * scoring requests". When nothing is promoted yet, we render an honest empty
 * state pointing at the forge.
 */
export function ProductionModelCard({
  model,
  onArchive,
  onTest,
  busy,
}: Props) {
  if (!model) {
    return (
      <section className="border-y border-hair border-rule py-12 text-center">
        <Anvil
          size={26}
          strokeWidth={1.2}
          aria-hidden
          className="mx-auto text-ink-40"
        />
        <p className="rubric mt-4">in service</p>
        <h3 className="display text-3xl mt-2">No production reranker.</h3>
        <p className="mt-3 text-sm text-ink-60 max-w-prose mx-auto">
          The retriever is currently running unranked &mdash; passages are returned
          purely on hybrid BM25 + vector fusion. Forge a model below and promote
          it to engage cross-encoder reranking.
        </p>
      </section>
    );
  }

  const promoted = model.promoted_at
    ? new Date(model.promoted_at).toLocaleString(undefined, {
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '—';

  return (
    <section
      className={cn(
        'panel-deep relative overflow-hidden',
        'p-8 lg:p-10',
        'before:absolute before:inset-0 before:pointer-events-none',
        'before:bg-[radial-gradient(circle_at_top_right,_rgba(140,40,30,0.08),_transparent_60%)]',
      )}
    >
      <div className="relative grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-8 items-start">
        <div className="min-w-0">
          <div className="flex items-baseline gap-3">
            <ShieldCheck
              size={14}
              strokeWidth={1.6}
              className="text-seal -mb-px"
              aria-hidden
            />
            <p className="rubric text-seal">in service · production</p>
          </div>
          <h3 className="display text-4xl lg:text-5xl mt-2 break-words">
            {model.name}
          </h3>
          <p className="datum text-2xs text-ink-60 uppercase tracking-rubric mt-2">
            {model.framework} · {model.backend} · {model.version}
          </p>
          <p className="text-sm text-ink-80 mt-5 max-w-prose leading-relaxed">
            Live for every inquiry &mdash; rescores the top retrieval candidates
            before they reach the answer agent. Promotion stamped {promoted}.
          </p>
        </div>

        <div className="flex flex-row lg:flex-col gap-3 shrink-0 self-stretch">
          {onTest && (
            <Button
              variant="outline"
              onClick={() => onTest(model.model_id)}
              disabled={busy}
              leftIcon={<Activity size={13} strokeWidth={1.6} />}
            >
              Bench-test
            </Button>
          )}
          {onArchive && (
            <Button
              variant="ghost"
              onClick={() => onArchive(model.model_id)}
              disabled={busy}
              leftIcon={<Clock size={13} strokeWidth={1.6} />}
            >
              Decommission
            </Button>
          )}
        </div>
      </div>

      <hr className="rule-soft my-7" />

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-5">
        <Metric label="holdout F1" value={model.holdout_f1} pct />
        <Metric label="ROC AUC" value={model.holdout_roc_auc} pct />
        <Metric
          label="separation"
          value={model.score_separation}
          digits={2}
        />
        <Metric label="rows trained" value={model.n_train} integer />
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  pct,
  integer,
  digits = 0,
}: {
  label: string;
  value: number;
  pct?: boolean;
  integer?: boolean;
  digits?: number;
}) {
  let display: string;
  if (integer) {
    display = value.toLocaleString();
  } else if (pct) {
    display = (value * 100).toFixed(digits);
  } else {
    display = value.toFixed(digits === 0 ? 2 : digits);
  }
  return (
    <div>
      <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
        {label}
      </p>
      <p className="display text-3xl mt-1 leading-none">
        {display}
        {pct && <span className="datum text-xs text-ink-40 ml-1">%</span>}
      </p>
    </div>
  );
}
