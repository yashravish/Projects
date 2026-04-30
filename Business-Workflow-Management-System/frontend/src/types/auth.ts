export type UserRole = 'analyst' | 'manager';

export interface User {
  id: number;
  name: string;
  email: string;
  role: UserRole;
  created_at: string;
  updated_at: string;
}

export interface UserSummary {
  id: number;
  name: string;
  email: string;
  role: UserRole;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}
