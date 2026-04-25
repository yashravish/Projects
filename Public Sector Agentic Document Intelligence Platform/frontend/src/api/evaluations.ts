import { paginationParams, request } from './client';
import {
  DatasetSchema,
  EvaluationRunDetailSchema,
  EvaluationRunListSchema,
  type EvalDataset,
  type EvaluationRunDetail,
  type EvaluationRunList,
} from './schemas';

/* High-level evaluation harness API. */

export interface EvaluationRunPayload {
  dataset_name?: string;
  top_k?: number;
  candidate_k?: number;
}

export async function getDataset(): Promise<EvalDataset> {
  return request(DatasetSchema, {
    url: '/api/v1/evaluations/dataset',
    method: 'GET',
  });
}

export async function postEvaluationRun(
  payload: EvaluationRunPayload = {},
): Promise<EvaluationRunDetail> {
  return request(EvaluationRunDetailSchema, {
    url: '/api/v1/evaluations/run',
    method: 'POST',
    data: payload,
    // Eval can take a few seconds with offline LLM and longer with a real one;
    // give it a generous ceiling so the UI doesn't false-positive a timeout.
    timeout: 120_000,
  });
}

export async function listEvaluationRuns(
  page = 1,
  pageSize = 25,
): Promise<EvaluationRunList> {
  return request(EvaluationRunListSchema, {
    url: '/api/v1/evaluations',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function getEvaluationRun(
  runId: string,
): Promise<EvaluationRunDetail> {
  return request(EvaluationRunDetailSchema, {
    url: `/api/v1/evaluations/${runId}`,
    method: 'GET',
  });
}
