import { request } from './client';
import {
  AccessTokenSchema,
  RegisterResponseSchema,
  TokenPairSchema,
  UserOutSchema,
} from './schemas';

export async function login(email: string, password: string) {
  return request(TokenPairSchema, {
    url: '/api/v1/auth/login',
    method: 'POST',
    data: { email, password },
  });
}

export async function register(
  email: string,
  password: string,
  organization_name: string,
) {
  return request(RegisterResponseSchema, {
    url: '/api/v1/auth/register',
    method: 'POST',
    data: { email, password, organization_name },
  });
}

export async function fetchMe() {
  return request(UserOutSchema, {
    url: '/api/v1/auth/me',
    method: 'GET',
  });
}

export async function refreshAccessToken(refresh_token: string) {
  return request(AccessTokenSchema, {
    url: '/api/v1/auth/refresh',
    method: 'POST',
    data: { refresh_token },
  });
}
