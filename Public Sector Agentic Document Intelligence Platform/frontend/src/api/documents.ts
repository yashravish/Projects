import { z } from 'zod';
import { paginationParams, request } from './client';
import {
  DocumentListSchema,
  DocumentOutSchema,
  HealthSchema,
  UploadResponseSchema,
} from './schemas';

export async function listDocuments(page = 1, pageSize = 50) {
  return request(DocumentListSchema, {
    url: '/api/v1/documents',
    method: 'GET',
    params: paginationParams(page, pageSize),
  });
}

export async function getDocument(id: string) {
  return request(DocumentOutSchema, {
    url: `/api/v1/documents/${id}`,
    method: 'GET',
  });
}

export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append('file', file);
  return request(UploadResponseSchema, {
    url: '/api/v1/documents/upload',
    method: 'POST',
    data: form,
    headers: { 'Content-Type': 'multipart/form-data' },
  });
}

export async function deleteDocument(id: string) {
  await request(z.unknown(), {
    url: `/api/v1/documents/${id}`,
    method: 'DELETE',
  });
}

export async function fetchHealth() {
  return request(HealthSchema, { url: '/health', method: 'GET' });
}
