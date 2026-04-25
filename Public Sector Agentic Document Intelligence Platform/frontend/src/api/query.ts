import { paginationParams, request } from './client';
import {
  InquiryResponseSchema,
  QueryRunListSchema,
  type InquiryResponse,
  type QueryRunList,
} from './schemas';

/* High-level inquiry API. Pydantic DTOs in `app.schemas.query` are the source
   of truth; the Zod mirror lives in `./schemas`. */

export interface InquiryPayload {
  question: string;
  top_k?: number;
  candidate_k?: number;
}

export async function postInquiry(payload: InquiryPayload): Promise<InquiryResponse> {
  return request(InquiryResponseSchema, {
    url: '/api/v1/query/inquiry',
    method: 'POST',
    data: payload,
  });
}

export async function listQueryRuns(
  page = 1,
  pageSize = 25,
): Promise<QueryRunList> {
  return request(QueryRunListSchema, {
    url: '/api/v1/query/runs',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function getQueryRun(runId: string): Promise<InquiryResponse> {
  return request(InquiryResponseSchema, {
    url: `/api/v1/query/runs/${runId}`,
    method: 'GET',
  });
}
