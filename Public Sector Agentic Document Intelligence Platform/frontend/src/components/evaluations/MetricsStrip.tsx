import type { AggregateMetrics } from '@/api/schemas';
import { MetricGauge } from './MetricGauge';

/**
 * The dashboard strip.
 *
 * Renders the aggregate metrics as a single-row instrument panel. Layout
 * is a 5-column grid on wide screens, wrapping to 2 columns on phones.
 */
export function MetricsStrip({ aggregate }: { aggregate: AggregateMetrics }) {
  return (
    <section
      aria-label="Aggregate metrics"
      className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-x-10 gap-y-6 border-y border-hair border-rule py-7"
    >
      <MetricGauge
        label="Pass rate"
        value={aggregate.pass_rate}
        threshold={0.7}
        hint={`${aggregate.n_items - aggregate.n_failures} / ${aggregate.n_items} passed`}
      />
      <MetricGauge
        label="Retrieval recall"
        value={aggregate.retrieval_recall}
        threshold={0.7}
      />
      <MetricGauge
        label="Faithfulness"
        value={aggregate.faithfulness}
        threshold={0.7}
        hint="phrases hit"
      />
      <MetricGauge
        label="Grounding"
        value={aggregate.grounding_score}
        threshold={0.7}
        hint="critic"
      />
      <MetricGauge
        label="Hallucination risk"
        value={aggregate.hallucination_risk}
        threshold={0.3}
        inverted
        hint="lower is better"
      />
      <MetricGauge
        label="Citation precision"
        value={aggregate.citation_precision}
        threshold={0.5}
      />
      <MetricGauge
        label="Citation recall"
        value={aggregate.citation_recall}
        threshold={0.5}
      />
      <MetricGauge
        label="Forbidden phrases"
        value={aggregate.forbidden_phrase_rate}
        threshold={0}
        inverted
        hint="zero tolerance"
      />
      <MetricGauge
        label="Latency p50"
        value={aggregate.latency_ms_p50}
        format="ms"
      />
      <MetricGauge
        label="Latency p95"
        value={aggregate.latency_ms_p95}
        format="ms"
      />
    </section>
  );
}
