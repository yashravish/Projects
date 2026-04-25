import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ScrollText, Search, Sparkle, RefreshCw } from 'lucide-react';
import {
  getQueryRun,
  listQueryRuns,
  postInquiry,
  type InquiryPayload,
} from '@/api/query';
import type { InquiryResponse } from '@/api/schemas';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { EmptyState } from '@/components/ui/EmptyState';
import { useToast } from '@/components/ui/Toast';
import { AgentTimeline } from '@/components/inquiry/AgentTimeline';
import { CitationCard } from '@/components/inquiry/CitationCard';
import { AnswerRender } from '@/components/inquiry/AnswerRender';
import { CritiqueStamp } from '@/components/inquiry/CritiqueStamp';
import { InquiryHistory } from '@/components/inquiry/InquiryHistory';
import { cn } from '@/lib/cn';

/** Agent inquiry: question, answer, evidence, trace, and re-loadable run history. */

const SUGGESTIONS: string[] = [
  'What changes did the FY24 procurement reform introduce?',
  'Summarise the data-retention obligations under §15.',
  'What budgetary impact does Annex C estimate for the next biennium?',
];

export function InquiryPage() {
  const qc = useQueryClient();
  const { push } = useToast();

  const [question, setQuestion] = useState('');
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [response, setResponse] = useState<InquiryResponse | null>(null);
  const [highlightedCitation, setHighlightedCitation] = useState<number | null>(
    null,
  );
  const answerRef = useRef<HTMLDivElement>(null);

  const history = useQuery({
    queryKey: ['query-runs'],
    queryFn: () => listQueryRuns(1, 25),
  });

  const detail = useMutation({
    mutationFn: (runId: string) => getQueryRun(runId),
    onSuccess: (resp) => {
      setResponse(resp);
      setActiveRunId(resp.run_id);
      setQuestion(resp.question);
      setTimeout(
        () => answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        50,
      );
    },
    onError: (err: unknown) => {
      push(err instanceof Error ? err.message : 'Could not load run', 'error');
    },
  });

  const inquiry = useMutation({
    mutationFn: (payload: InquiryPayload) => postInquiry(payload),
    onMutate: () => {
      setResponse(null);
      setHighlightedCitation(null);
      setActiveRunId(null);
    },
    onSuccess: (resp) => {
      setResponse(resp);
      setActiveRunId(resp.run_id);
      void qc.invalidateQueries({ queryKey: ['query-runs'] });
      if (resp.status === 'failed') {
        push(resp.error ?? 'The agent could not produce an answer.', 'error');
      } else if (!resp.critique.passed) {
        push(
          'Answer returned, but the critic flagged validation issues — review the evidence.',
          'info',
        );
      } else {
        push('Inquiry validated. Answer filed.', 'success');
      }
      setTimeout(
        () => answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        50,
      );
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : 'Inquiry failed';
      push(msg, 'error');
    },
  });

  const trimmed = question.trim();
  const tooShort = trimmed.length < 6;
  const running = inquiry.isPending;

  function submit() {
    if (running || tooShort) return;
    inquiry.mutate({ question: trimmed });
  }

  function retry() {
    if (!response) return;
    inquiry.mutate({ question: response.question });
  }

  return (
    <div className="stagger">
      <header className="mb-10">
        <p className="rubric">003 — The Inquiry Desk</p>
        <h1 className="display text-5xl mt-2">Pose a question.</h1>
        <p className="mt-4 text-base text-ink-80 max-w-prose leading-relaxed">
          The agent will plan, retrieve from your filed corpus, synthesise an
          answer with inline citations, and submit that answer to a critic for
          grounding and hallucination scoring before it reaches you.
        </p>
      </header>

      <PromptForm
        question={question}
        onChange={setQuestion}
        onSubmit={submit}
        running={running}
        tooShort={tooShort}
      />

      {!response && !running && trimmed.length === 0 && (
        <Suggestions onPick={setQuestion} />
      )}

      <hr className="rule-double my-10" />

      {/* Process column. */}
      <section className="mb-10">
        <p className="rubric mb-3">003.1 — Process trace</p>
        <AgentTimeline trace={response?.trace ?? []} running={running} />
      </section>

      {/* Body — answer + evidence rail. */}
      <div ref={answerRef} className="grid-dossier">
        <aside>
          <p className="rubric mb-3">003.3 — Evidence on file</p>
          {response ? (
            response.citations.length === 0 ? (
              <p className="text-sm text-ink-60 leading-relaxed">
                No citation chunks were retained. The agent declined to ground
                its answer.
              </p>
            ) : (
              <div className="border-t border-hair border-rule">
                {response.citations.map((c) => (
                  <CitationCard
                    key={`${c.chunk_id}-${c.index}`}
                    citation={c}
                    highlighted={highlightedCitation === c.index}
                    onClick={() => setHighlightedCitation(c.index)}
                  />
                ))}
              </div>
            )
          ) : (
            <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
              No evidence in scope.
            </p>
          )}
        </aside>

        <section>
          <p className="rubric mb-3">003.2 — Answer of record</p>

          {inquiry.isError ? (
            <ErrorState
              title="The agent failed."
              description={
                inquiry.error instanceof Error
                  ? inquiry.error.message
                  : 'Unknown error'
              }
              action={
                <Button onClick={retry} leftIcon={<RefreshCw size={14} />}>
                  Retry
                </Button>
              }
            />
          ) : running ? (
            <Skeleton rows={6} />
          ) : !response ? (
            <EmptyState
              rubric="awaiting prompt"
              title="No inquiry submitted yet."
              description="Type your question above. Inquiries are persisted to your tenant's audit ledger; you can revisit any past run from the roll below."
            />
          ) : response.status === 'failed' ? (
            <ErrorState
              title="The pipeline failed."
              description={response.error ?? 'No further detail.'}
              action={
                <Button onClick={retry} leftIcon={<RefreshCw size={14} />}>
                  Retry
                </Button>
              }
            />
          ) : (
            <article className="space-y-6">
              <CritiqueStamp critique={response.critique} />
              <AnswerRender
                text={response.answer_text}
                citations={response.citations}
                highlightedIndex={highlightedCitation}
                onCitationClick={(i) =>
                  setHighlightedCitation((cur) => (cur === i ? null : i))
                }
              />
              <RunMeta resp={response} />
            </article>
          )}
        </section>
      </div>

      <hr className="rule-double my-12" />

      <section>
        <header className="flex items-baseline justify-between mb-5">
          <div>
            <p className="rubric">003.4 — Past inquiries</p>
            <h2 className="display text-3xl mt-1">The roll.</h2>
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
          <Skeleton rows={4} />
        ) : history.isError ? (
          <ErrorState
            title="Could not load past inquiries"
            description={
              history.error instanceof Error ? history.error.message : 'Unknown error'
            }
            action={<Button onClick={() => history.refetch()}>Retry</Button>}
          />
        ) : (
          <InquiryHistory
            items={history.data?.items ?? []}
            activeRunId={activeRunId}
            onSelect={(id) => detail.mutate(id)}
          />
        )}
      </section>
    </div>
  );
}

interface PromptFormProps {
  question: string;
  onChange: (s: string) => void;
  onSubmit: () => void;
  running: boolean;
  tooShort: boolean;
}

function PromptForm({
  question,
  onChange,
  onSubmit,
  running,
  tooShort,
}: PromptFormProps) {
  const ref = useRef<HTMLTextAreaElement>(null);
  // Auto-grow the textarea up to ~6 lines.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = '0px';
    const cap = 8.5 * 16; // ~8 lines
    el.style.height = `${Math.min(el.scrollHeight, cap)}px`;
  }, [question]);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className={cn(
        'panel-deep border-hair border-rule-soft px-6 py-5',
        'transition-colors duration-200',
      )}
    >
      <label className="block">
        <span className="rubric flex items-center gap-2">
          <ScrollText size={12} aria-hidden /> question
        </span>
        <textarea
          ref={ref}
          rows={2}
          value={question}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              onSubmit();
            }
          }}
          placeholder="What does the agent need to find?"
          className={cn(
            'mt-2 w-full resize-none bg-transparent border-0 outline-none',
            'display text-2xl leading-snug placeholder:text-ink-40',
            'focus:bg-transparent',
          )}
          aria-label="Inquiry question"
        />
      </label>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
          Cmd/Ctrl + Enter to submit
        </p>
        <div className="flex items-center gap-3">
          {tooShort && question.length > 0 && (
            <span className="datum text-2xs text-seal uppercase tracking-rubric">
              minimum 6 characters
            </span>
          )}
          <Button
            type="submit"
            loading={running}
            disabled={tooShort}
            leftIcon={<Search size={14} />}
          >
            File inquiry
          </Button>
        </div>
      </div>
    </form>
  );
}

function Suggestions({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      <span className="datum text-2xs text-ink-40 uppercase tracking-rubric mr-2 mt-1.5">
        Try
      </span>
      {SUGGESTIONS.map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => onPick(s)}
          className={cn(
            'inline-flex items-center gap-1.5 px-3 py-1.5 border-hair border-rule-soft',
            'text-xs text-ink-80 transition-colors duration-200',
            'hover:border-seal hover:text-seal hover:bg-seal/5',
          )}
        >
          <Sparkle size={11} strokeWidth={1.5} aria-hidden />
          <span>{s}</span>
        </button>
      ))}
    </div>
  );
}

function RunMeta({ resp }: { resp: InquiryResponse }) {
  return (
    <dl className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-3 border-t border-hair border-rule-soft pt-4">
      <Pair label="model" value={resp.model} />
      <Pair label="latency" value={`${resp.latency_ms} ms`} />
      <Pair label="tokens in/out" value={`${resp.token_input} / ${resp.token_output}`} />
      <Pair label="cost" value={`$${resp.cost_usd.toFixed(4)}`} />
      <Pair label="citations" value={`${resp.citations.length}`} />
      <Pair label="retrieved" value={`${resp.retrieved.length}`} />
      {resp.mlflow_run_id && (
        <Pair label="mlflow run" value={resp.mlflow_run_id.slice(0, 8) + '…'} />
      )}
      <Pair
        label="filed"
        value={new Date(resp.created_at).toLocaleString(undefined, {
          month: 'short',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
        })}
      />
    </dl>
  );
}

function Pair({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="datum text-2xs text-ink-60 uppercase tracking-rubric">
        {label}
      </dt>
      <dd className="datum text-sm text-ink mt-0.5 truncate">{value}</dd>
    </div>
  );
}
