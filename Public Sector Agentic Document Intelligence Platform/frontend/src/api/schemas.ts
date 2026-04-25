import { z } from 'zod';

/* Zod mirror of Pydantic DTOs under `app.schemas.*`. Pydantic is the API
   source of truth; these shapes validate client-side JSON and narrow types. */

export const OrganizationOutSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  slug: z.string(),
  created_at: z.string(),
});
export type Organization = z.infer<typeof OrganizationOutSchema>;

export const UserOutSchema = z.object({
  id: z.string().uuid(),
  organization_id: z.string().uuid(),
  email: z.string().email(),
  role: z.enum(['admin', 'analyst', 'viewer']),
  is_active: z.boolean(),
  created_at: z.string(),
});
export type User = z.infer<typeof UserOutSchema>;

export const TokenPairSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal('bearer'),
});
export type TokenPair = z.infer<typeof TokenPairSchema>;

export const AccessTokenSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer'),
});
export type AccessToken = z.infer<typeof AccessTokenSchema>;

/** POST `/auth/refresh` — Pydantic `AccessToken` plus optional `expires_in` if the backend adds it later. */
export const AuthRefreshResponseSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer'),
  expires_in: z.number().int().nonnegative().optional(),
});
export type AuthRefreshResponse = z.infer<typeof AuthRefreshResponseSchema>;

export const RegisterResponseSchema = z.object({
  user: UserOutSchema,
  organization: OrganizationOutSchema,
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal('bearer'),
});
export type RegisterResponse = z.infer<typeof RegisterResponseSchema>;

export const DocumentStatusSchema = z.enum([
  'pending',
  'extracting',
  'chunking',
  'embedding',
  'ready',
  'failed',
]);
export type DocumentStatus = z.infer<typeof DocumentStatusSchema>;

export const DocumentListItemSchema = z.object({
  id: z.string().uuid(),
  filename: z.string(),
  page_count: z.number().int().nonnegative(),
  byte_size: z.number().int().nonnegative(),
  status: DocumentStatusSchema,
  chunk_count: z.number().int().nonnegative(),
  created_at: z.string(),
});
export type DocumentListItem = z.infer<typeof DocumentListItemSchema>;

export const DocumentListSchema = z.object({
  items: z.array(DocumentListItemSchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type DocumentList = z.infer<typeof DocumentListSchema>;

export const DocumentOutSchema = DocumentListItemSchema.extend({
  organization_id: z.string().uuid(),
  uploaded_by: z.string().uuid().nullable(),
  sha256: z.string(),
  error_message: z.string().nullable(),
  updated_at: z.string(),
});
export type Document = z.infer<typeof DocumentOutSchema>;

export const UploadResponseSchema = z.object({
  document_id: z.string().uuid(),
  status: DocumentStatusSchema,
  duplicate: z.boolean(),
});
export type UploadResponse = z.infer<typeof UploadResponseSchema>;

const HealthComponentStatusSchema = z.enum([
  'ok',
  'degraded',
  'down',
  'not_configured',
]);

export const HealthSchema = z.object({
  status: z.enum(['ok', 'degraded']),
  db: HealthComponentStatusSchema,
  redis: HealthComponentStatusSchema,
  mlflow: HealthComponentStatusSchema,
  openai: HealthComponentStatusSchema,
});
export type Health = z.infer<typeof HealthSchema>;

// Query / inquiry

export const CitationSchema = z.object({
  index: z.number().int().positive(),
  chunk_id: z.string().uuid(),
  document_id: z.string().uuid(),
  document_filename: z.string(),
  page_start: z.number().int().nonnegative(),
  page_end: z.number().int().nonnegative(),
  snippet: z.string(),
});
export type Citation = z.infer<typeof CitationSchema>;

export const RetrievedChunkSchema = z.object({
  chunk_id: z.string().uuid(),
  document_id: z.string().uuid(),
  document_filename: z.string(),
  page_start: z.number().int().nonnegative(),
  page_end: z.number().int().nonnegative(),
  chunk_index: z.number().int().nonnegative(),
  fused_score: z.number(),
  bm25_rank: z.number().int(),
  bm25_score: z.number(),
  vector_rank: z.number().int(),
  vector_score: z.number(),
  snippet: z.string(),
});
export type RetrievedChunk = z.infer<typeof RetrievedChunkSchema>;

export const TraceStepSchema = z.object({
  node: z.string(),
  label: z.string(),
  detail: z.string(),
  duration_ms: z.number().int().nonnegative(),
  metadata: z.record(z.unknown()).default({}),
});
export type TraceStep = z.infer<typeof TraceStepSchema>;

export const CritiqueSchema = z.object({
  grounding_score: z.number().min(0).max(1),
  hallucination_risk: z.number().min(0).max(1),
  passed: z.boolean(),
  issues: z.array(z.string()),
});
export type Critique = z.infer<typeof CritiqueSchema>;

export const QueryStatusSchema = z.enum(['success', 'failed']);
export type QueryStatus = z.infer<typeof QueryStatusSchema>;

export const InquiryResponseSchema = z.object({
  run_id: z.string().uuid(),
  status: QueryStatusSchema,
  question: z.string(),
  answer_text: z.string(),
  citations: z.array(CitationSchema),
  retrieved: z.array(RetrievedChunkSchema),
  critique: CritiqueSchema,
  trace: z.array(TraceStepSchema),
  model: z.string(),
  latency_ms: z.number().int().nonnegative(),
  token_input: z.number().int().nonnegative(),
  token_output: z.number().int().nonnegative(),
  cost_usd: z.number().nonnegative(),
  mlflow_run_id: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  created_at: z.string(),
});
export type InquiryResponse = z.infer<typeof InquiryResponseSchema>;

export const QueryRunListItemSchema = z.object({
  run_id: z.string().uuid(),
  question: z.string(),
  status: QueryStatusSchema,
  grounding_score: z.number().nullable(),
  hallucination_risk: z.number().nullable(),
  n_citations: z.number().int().nonnegative(),
  latency_ms: z.number().int().nonnegative(),
  model: z.string(),
  created_at: z.string(),
});
export type QueryRunListItem = z.infer<typeof QueryRunListItemSchema>;

export const QueryRunListSchema = z.object({
  items: z.array(QueryRunListItemSchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type QueryRunList = z.infer<typeof QueryRunListSchema>;

// Evaluation harness

export const GoldItemSchema = z.object({
  id: z.string(),
  question: z.string(),
  expected_doc_filenames: z.array(z.string()),
  must_contain_any: z.array(z.array(z.string())),
  forbidden_phrases: z.array(z.string()),
  topic: z.string(),
});
export type GoldItem = z.infer<typeof GoldItemSchema>;

export const DatasetSchema = z.object({
  name: z.string(),
  description: z.string(),
  version: z.string(),
  n_items: z.number().int().nonnegative(),
  items: z.array(GoldItemSchema),
});
export type EvalDataset = z.infer<typeof DatasetSchema>;

export const ItemMetricsSchema = z.object({
  item_id: z.string(),
  retrieval_recall: z.number(),
  retrieval_precision: z.number(),
  citation_precision: z.number(),
  citation_recall: z.number(),
  faithfulness: z.number(),
  forbidden_phrase_rate: z.number(),
  grounding_score: z.number(),
  hallucination_risk: z.number(),
  answer_passed_critic: z.boolean(),
  item_passed: z.boolean(),
  latency_ms: z.number().int().nonnegative(),
  n_retrieved: z.number().int().nonnegative(),
  n_citations: z.number().int().nonnegative(),
});
export type ItemMetrics = z.infer<typeof ItemMetricsSchema>;

export const AggregateMetricsSchema = z.object({
  n_items: z.number().int().nonnegative(),
  pass_rate: z.number(),
  retrieval_recall: z.number(),
  retrieval_precision: z.number(),
  citation_precision: z.number(),
  citation_recall: z.number(),
  faithfulness: z.number(),
  forbidden_phrase_rate: z.number(),
  grounding_score: z.number(),
  hallucination_risk: z.number(),
  latency_ms_p50: z.number(),
  latency_ms_p95: z.number(),
  n_failures: z.number().int().nonnegative(),
});
export type AggregateMetrics = z.infer<typeof AggregateMetricsSchema>;

export const EvalCitationSchema = z.object({
  document_filename: z.string(),
  page_start: z.number().int().nonnegative(),
  page_end: z.number().int().nonnegative(),
  snippet: z.string(),
});
export type EvalCitation = z.infer<typeof EvalCitationSchema>;

export const EvalItemAnswerSchema = z.object({
  question: z.string(),
  answer_text: z.string(),
  error: z.string().nullable().optional(),
  citations: z.array(EvalCitationSchema),
  grounding_score: z.number(),
  hallucination_risk: z.number(),
  passed: z.boolean(),
  latency_ms: z.number().int().nonnegative(),
  cost_usd: z.number(),
});
export type EvalItemAnswer = z.infer<typeof EvalItemAnswerSchema>;

export const EvaluationItemSchema = z.object({
  gold: GoldItemSchema,
  metrics: ItemMetricsSchema,
  inquiry: EvalItemAnswerSchema,
});
export type EvaluationItem = z.infer<typeof EvaluationItemSchema>;

export const EvaluationStatusSchema = z.enum([
  'pending',
  'running',
  'success',
  'failed',
]);
export type EvaluationStatus = z.infer<typeof EvaluationStatusSchema>;

export const EvaluationRunSummarySchema = z.object({
  run_id: z.string().uuid(),
  dataset_name: z.string(),
  dataset_version: z.string(),
  model: z.string(),
  status: EvaluationStatusSchema,
  n_items: z.number().int().nonnegative(),
  pass_rate: z.number(),
  grounding_score: z.number(),
  faithfulness: z.number(),
  retrieval_recall: z.number(),
  latency_ms_p50: z.number(),
  mlflow_run_id: z.string().nullable().optional(),
  created_at: z.string(),
});
export type EvaluationRunSummary = z.infer<typeof EvaluationRunSummarySchema>;

export const EvaluationRunListSchema = z.object({
  items: z.array(EvaluationRunSummarySchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type EvaluationRunList = z.infer<typeof EvaluationRunListSchema>;

export const EvaluationRunDetailSchema = z.object({
  run_id: z.string().uuid(),
  dataset_name: z.string(),
  dataset_version: z.string(),
  model: z.string(),
  status: EvaluationStatusSchema,
  aggregate: AggregateMetricsSchema,
  items: z.array(EvaluationItemSchema),
  prompt_versions: z.record(z.unknown()),
  retrieval_config: z.record(z.unknown()),
  wall_time_ms: z.number().int().nonnegative(),
  mlflow_run_id: z.string().nullable().optional(),
  created_at: z.string(),
});
export type EvaluationRunDetail = z.infer<typeof EvaluationRunDetailSchema>;

// Audit & governance (mirrors backend `app.schemas.audit`)

export const AuditOutcomeSchema = z.enum(['success', 'denied', 'error']);
export type AuditOutcome = z.infer<typeof AuditOutcomeSchema>;

export const RetentionResourceSchema = z.enum([
  'document',
  'query_run',
  'evaluation_run',
]);
export type RetentionResource = z.infer<typeof RetentionResourceSchema>;

export const RetentionStatusSchema = z.enum(['running', 'success', 'failed']);
export type RetentionStatus = z.infer<typeof RetentionStatusSchema>;

export const AuditEventSchema = z.object({
  event_id: z.string().uuid(),
  organization_id: z.string().uuid(),
  actor_id: z.string().uuid().nullable(),
  actor_email: z.string().nullable(),
  action: z.string(),
  resource_type: z.string(),
  resource_id: z.string().uuid().nullable(),
  outcome: AuditOutcomeSchema,
  request_id: z.string().nullable(),
  prev_hash: z.string().nullable(),
  entry_hash: z.string(),
  metadata: z.record(z.unknown()).default({}),
  created_at: z.string(),
});
export type AuditEvent = z.infer<typeof AuditEventSchema>;

export const AuditEventListSchema = z.object({
  items: z.array(AuditEventSchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type AuditEventList = z.infer<typeof AuditEventListSchema>;

export const IntegrityBreakSchema = z.object({
  event_id: z.string().uuid(),
  expected_prev_hash: z.string().nullable(),
  observed_prev_hash: z.string().nullable(),
  expected_entry_hash: z.string(),
  observed_entry_hash: z.string(),
  created_at: z.string(),
});
export type IntegrityBreak = z.infer<typeof IntegrityBreakSchema>;

export const IntegrityReportSchema = z.object({
  organization_id: z.string().uuid(),
  verified_at: z.string(),
  total_events: z.number().int().nonnegative(),
  chain_ok: z.boolean(),
  head_hash: z.string().nullable(),
  tail_hash: z.string().nullable(),
  breaks: z.array(IntegrityBreakSchema).default([]),
});
export type IntegrityReport = z.infer<typeof IntegrityReportSchema>;

export const LedgerStatsSchema = z.object({
  total_events: z.number().int().nonnegative(),
  events_24h: z.number().int().nonnegative(),
  events_7d: z.number().int().nonnegative(),
  distinct_actions: z.number().int().nonnegative(),
  distinct_actors: z.number().int().nonnegative(),
  last_event_at: z.string().nullable(),
  head_hash: z.string().nullable(),
  tail_hash: z.string().nullable(),
});
export type LedgerStats = z.infer<typeof LedgerStatsSchema>;

export const RetentionPolicySchema = z.object({
  policy_id: z.string().uuid(),
  resource_type: RetentionResourceSchema,
  ttl_days: z.number().int().nonnegative(),
  is_active: z.boolean(),
  notes: z.string().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
});
export type RetentionPolicy = z.infer<typeof RetentionPolicySchema>;

export const RetentionPolicyListSchema = z.object({
  items: z.array(RetentionPolicySchema),
});
export type RetentionPolicyList = z.infer<typeof RetentionPolicyListSchema>;

export const RetentionRunSchema = z.object({
  run_id: z.string().uuid(),
  triggered_by: z.string().uuid().nullable(),
  status: RetentionStatusSchema,
  purged_counts: z.record(z.number().int().nonnegative()),
  error_message: z.string().nullable(),
  started_at: z.string(),
  finished_at: z.string().nullable(),
});
export type RetentionRun = z.infer<typeof RetentionRunSchema>;

export const RetentionRunListSchema = z.object({
  items: z.array(RetentionRunSchema),
  total: z.number().int().nonnegative(),
});
export type RetentionRunList = z.infer<typeof RetentionRunListSchema>;

// Training & model registry

export const TrainingJobStatusSchema = z.enum([
  'pending',
  'running',
  'success',
  'failed',
]);
export type TrainingJobStatus = z.infer<typeof TrainingJobStatusSchema>;

export const ModelStageSchema = z.enum(['staging', 'production', 'archived']);
export type ModelStage = z.infer<typeof ModelStageSchema>;

export const TrainingJobMetricsSchema = z
  .object({
    n_train: z.number().default(0),
    n_holdout: z.number().default(0),
    train_accuracy: z.number().default(0),
    holdout_accuracy: z.number().default(0),
    holdout_precision: z.number().default(0),
    holdout_recall: z.number().default(0),
    holdout_f1: z.number().default(0),
    holdout_roc_auc: z.number().default(0),
    holdout_avg_precision: z.number().default(0),
    holdout_log_loss: z.number().default(0),
    score_separation: z.number().default(0),
  })
  .catchall(z.unknown());
export type TrainingJobMetrics = z.infer<typeof TrainingJobMetricsSchema>;

export const TrainingJobSummarySchema = z.object({
  job_id: z.string().uuid(),
  name: z.string(),
  version: z.string(),
  backend: z.string(),
  framework: z.string(),
  status: TrainingJobStatusSchema,
  duration_s: z.number(),
  holdout_f1: z.number().default(0),
  holdout_roc_auc: z.number().default(0),
  score_separation: z.number().default(0),
  n_train: z.number().int().nonnegative().default(0),
  error_message: z.string().nullable().optional(),
  mlflow_run_id: z.string().nullable().optional(),
  created_at: z.string(),
});
export type TrainingJobSummary = z.infer<typeof TrainingJobSummarySchema>;

export const TrainingJobListSchema = z.object({
  items: z.array(TrainingJobSummarySchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type TrainingJobList = z.infer<typeof TrainingJobListSchema>;

export const TrainingJobDetailSchema = z.object({
  job_id: z.string().uuid(),
  organization_id: z.string().uuid(),
  triggered_by: z.string().uuid().nullable(),
  name: z.string(),
  version: z.string(),
  backend: z.string(),
  framework: z.string(),
  framework_version: z.string().nullable().optional(),
  status: TrainingJobStatusSchema,
  artifact_uri: z.string().nullable().optional(),
  external_job_id: z.string().nullable().optional(),
  config: z.record(z.unknown()),
  metrics: TrainingJobMetricsSchema,
  manifest: z.record(z.unknown()),
  log_excerpt: z.string().nullable().optional(),
  duration_s: z.number(),
  mlflow_run_id: z.string().nullable().optional(),
  error_message: z.string().nullable().optional(),
  started_at: z.string().nullable().optional(),
  finished_at: z.string().nullable().optional(),
  created_at: z.string(),
  registered_model_id: z.string().uuid().nullable().optional(),
});
export type TrainingJobDetail = z.infer<typeof TrainingJobDetailSchema>;

export const RegisteredModelSummarySchema = z.object({
  model_id: z.string().uuid(),
  name: z.string(),
  version: z.string(),
  framework: z.string(),
  backend: z.string(),
  stage: ModelStageSchema,
  holdout_f1: z.number().default(0),
  holdout_roc_auc: z.number().default(0),
  score_separation: z.number().default(0),
  n_train: z.number().int().nonnegative().default(0),
  artifact_uri: z.string(),
  training_job_id: z.string().uuid().nullable(),
  created_at: z.string(),
  promoted_at: z.string().nullable(),
  archived_at: z.string().nullable(),
});
export type RegisteredModelSummary = z.infer<typeof RegisteredModelSummarySchema>;

export const RegisteredModelListSchema = z.object({
  items: z.array(RegisteredModelSummarySchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
});
export type RegisteredModelList = z.infer<typeof RegisteredModelListSchema>;

export const RegisteredModelDetailSchema = z.object({
  model_id: z.string().uuid(),
  organization_id: z.string().uuid(),
  name: z.string(),
  version: z.string(),
  framework: z.string(),
  framework_version: z.string().nullable().optional(),
  backend: z.string(),
  artifact_uri: z.string(),
  local_dir: z.string().nullable().optional(),
  stage: ModelStageSchema,
  metrics: TrainingJobMetricsSchema,
  manifest: z.record(z.unknown()),
  training_job_id: z.string().uuid().nullable(),
  promoted_by: z.string().uuid().nullable(),
  notes: z.string().nullable().optional(),
  promoted_at: z.string().nullable().optional(),
  archived_at: z.string().nullable().optional(),
  created_at: z.string(),
});
export type RegisteredModelDetail = z.infer<typeof RegisteredModelDetailSchema>;

export const ScoredPassageSchema = z.object({
  index: z.number().int(),
  passage: z.string(),
  score: z.number(),
});
export type ScoredPassage = z.infer<typeof ScoredPassageSchema>;

export const RerankerPredictResponseSchema = z.object({
  model_id: z.string().uuid(),
  model_name: z.string(),
  model_version: z.string(),
  backend: z.string(),
  scored: z.array(ScoredPassageSchema),
});
export type RerankerPredictResponse = z.infer<
  typeof RerankerPredictResponseSchema
>;
