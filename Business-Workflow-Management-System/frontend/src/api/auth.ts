import type { TokenResponse, User, UserSummary } from '../types/auth';
import { apiRequest } from './client';

export function login(email: string, password: string): Promise<TokenResponse> {
  return apiRequest<TokenResponse>('/auth/login', {
    method: 'POST',
    body: { email, password },
  });
}

export function fetchMe(): Promise<User> {
  return apiRequest<User>('/auth/me');
}

export function fetchUsers(): Promise<UserSummary[]> {
  return apiRequest<UserSummary[]>('/users');
}
