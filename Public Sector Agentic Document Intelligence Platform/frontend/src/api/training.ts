import { paginationParams, request } from './client';
import {
  RegisteredModelDetailSchema,
  RegisteredModelListSchema,
  RerankerPredictResponseSchema,
  TrainingJobDetailSchema,
  TrainingJobListSchema,
  type RegisteredModelDetail,
  type RegisteredModelList,
  type RerankerPredictResponse,
  type TrainingJobDetail,
  type TrainingJobList,
} from './schemas';

/* Training & model registry API. */

export interface SubmitTrainingPayload {
  name?: string;
  notes?: string | null;
  auto_promote?: boolean;
}

export async function submitTrainingJob(
  payload: SubmitTrainingPayload = {},
): Promise<TrainingJobDetail> {
  return request(TrainingJobDetailSchema, {
    url: '/api/v1/training/jobs',
    method: 'POST',
    data: {
      name: payload.name ?? 'psdi-cross-encoder-reranker',
      notes: payload.notes ?? null,
      auto_promote: payload.auto_promote ?? false,
    },
    /* Training is sync from the SPA's perspective. The local backend takes
       a few seconds; SageMaker can take minutes. Give it generous headroom
       so the UI doesn't surface a phantom timeout. */
    timeout: 600_000,
  });
}

export async function listTrainingJobs(
  page = 1,
  pageSize = 25,
): Promise<TrainingJobList> {
  return request(TrainingJobListSchema, {
    url: '/api/v1/training/jobs',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function getTrainingJob(jobId: string): Promise<TrainingJobDetail> {
  return request(TrainingJobDetailSchema, {
    url: `/api/v1/training/jobs/${jobId}`,
    method: 'GET',
  });
}

export async function listRegisteredModels(
  page = 1,
  pageSize = 50,
): Promise<RegisteredModelList> {
  return request(RegisteredModelListSchema, {
    url: '/api/v1/models',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function getRegisteredModel(
  modelId: string,
): Promise<RegisteredModelDetail> {
  return request(RegisteredModelDetailSchema, {
    url: `/api/v1/models/${modelId}`,
    method: 'GET',
  });
}

export interface PromoteModelPayload {
  stage: 'production' | 'archived';
  notes?: string | null;
}

export async function promoteModel(
  modelId: string,
  payload: PromoteModelPayload,
): Promise<RegisteredModelDetail> {
  return request(RegisteredModelDetailSchema, {
    url: `/api/v1/models/${modelId}/promote`,
    method: 'POST',
    data: { stage: payload.stage, notes: payload.notes ?? null },
  });
}

export interface RerankerPredictPayload {
  query: string;
  passages: string[];
}

export async function predictWithModel(
  modelId: string,
  payload: RerankerPredictPayload,
): Promise<RerankerPredictResponse> {
  return request(RerankerPredictResponseSchema, {
    url: `/api/v1/models/${modelId}/predict`,
    method: 'POST',
    data: payload,
  });
}
