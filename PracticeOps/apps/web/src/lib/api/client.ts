/**
 * API Client - Typed wrappers for all backend endpoints
 *
 * CONTRACT LOCK PROTOCOL:
 * - UI calls ONLY these client functions
 * - If endpoint mismatch: fix backend, then regenerate client in same patch
 * - All types must match backend OpenAPI spec
 */

import type {
  AcceptInviteRequest,
  AcceptInviteResponse,
  ActiveCycleResponse,
  ApiError,
  AssignmentsListResponse,
  AuthResponse,
  ClaimTicketResponse,
  CreateInviteRequest,
  CreateInviteResponse,
  CreatePracticeLogRequest,
  CreatePracticeLogResponse,
  CreateTicketRequest,
  CreateTicketResponse,
  CycleResponse,
  ComplianceInsightsResponse,
  InvitePreviewResponse,
  InvitesListResponse,
  LeaderDashboardResponse,
  LoginRequest,
  MeResponse,
  MemberDashboardResponse,
  MembersListResponse,
  PracticeLogsListResponse,
  RefreshResponse,
  RegisterRequest,
  RevokeInviteResponse,
  TicketActivitiesResponse,
  TicketsListResponse,
  TransitionTicketRequest,
  TransitionTicketResponse,
  UpdateMemberRequest,
  UpdateMemberResponse,
  UpdateTeamRequest,
  UpdateTeamResponse,
  UpdateTicketRequest,
  UpdateTicketResponse,
  VerifyTicketRequest,
  VerifyTicketResponse,
} from "./types";

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// =============================================================================
// Token Storage
// =============================================================================

const TOKEN_KEYS = {
  ACCESS: "practiceops_access_token",
  REFRESH: "practiceops_refresh_token",
} as const;

export function getAccessToken(): string | null {
  return localStorage.getItem(TOKEN_KEYS.ACCESS);
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(TOKEN_KEYS.REFRESH);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(TOKEN_KEYS.ACCESS, accessToken);
  localStorage.setItem(TOKEN_KEYS.REFRESH, refreshToken);
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_KEYS.ACCESS);
  localStorage.removeItem(TOKEN_KEYS.REFRESH);
}

// =============================================================================
// API Error Handling
// =============================================================================

export class ApiClientError extends Error {
  public readonly code: string;
  public readonly field: string | null;
  public readonly status: number;

  constructor(error: ApiError["error"], status: number) {
    super(error.message);
    this.name = "ApiClientError";
    this.code = error.code;
    this.field = error.field;
    this.status = status;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData: ApiError;
    try {
      errorData = await response.json();
    } catch {
      throw new ApiClientError(
        {
          code: "INTERNAL",
          message: "An unexpected error occurred",
          field: null,
        },
        response.status
      );
    }
    throw new ApiClientError(errorData.error, response.status);
  }
  return response.json();
}

// =============================================================================
// Request Helpers
// =============================================================================

type RequestOptions = {
  method: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  auth?: boolean;
};

async function request<T>(
  endpoint: string,
  options: RequestOptions
): Promise<T> {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  if (options.auth) {
    const token = getAccessToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: options.method,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  // Handle 401 with token refresh
  if (response.status === 401 && options.auth) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      // Retry the request with new token
      headers["Authorization"] = `Bearer ${getAccessToken()}`;
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: options.method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
      });
      return handleResponse<T>(retryResponse);
    }
  }

  return handleResponse<T>(response);
}

async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    clearTokens();
    return false;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      clearTokens();
      return false;
    }

    const data: RefreshResponse = await response.json();
    localStorage.setItem(TOKEN_KEYS.ACCESS, data.access_token);
    return true;
  } catch {
    clearTokens();
    return false;
  }
}

// =============================================================================
// Auth Endpoints
// =============================================================================

export async function register(data: RegisterRequest): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    body: data,
  });
}

export async function login(data: LoginRequest): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: data,
  });
}

export async function refreshToken(
  refresh_token: string
): Promise<RefreshResponse> {
  return request<RefreshResponse>("/auth/refresh", {
    method: "POST",
    body: { refresh_token },
  });
}

export async function getMe(): Promise<MeResponse> {
  return request<MeResponse>("/me", {
    method: "GET",
    auth: true,
  });
}

// =============================================================================
// Invite Endpoints
// =============================================================================

export async function getInvitePreview(
  token: string
): Promise<InvitePreviewResponse> {
  return request<InvitePreviewResponse>(`/invites/${token}`, {
    method: "GET",
  });
}

export async function acceptInvite(
  token: string,
  data: AcceptInviteRequest = {}
): Promise<AcceptInviteResponse> {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  // Include auth header if logged in
  const accessToken = getAccessToken();
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(`${API_BASE_URL}/invites/${token}/accept`, {
    method: "POST",
    headers,
    body: JSON.stringify(data),
  });

  return handleResponse<AcceptInviteResponse>(response);
}

// =============================================================================
// Cycle Endpoints
// =============================================================================

export async function getActiveCycle(
  teamId: string
): Promise<ActiveCycleResponse> {
  return request<ActiveCycleResponse>(`/teams/${teamId}/cycles/active`, {
    method: "GET",
    auth: true,
  });
}

export interface CreateCycleRequest {
  date: string; // ISO date string
  name?: string;
}

export interface CreateCycleResponse {
  cycle: CycleResponse;
}

export interface CyclesListResponse {
  items: CycleResponse[];
  next_cursor: string | null;
}

export async function createCycle(
  teamId: string,
  data: CreateCycleRequest
): Promise<CreateCycleResponse> {
  return request<CreateCycleResponse>(`/teams/${teamId}/cycles`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export async function listCycles(
  teamId: string,
  cursor?: string
): Promise<CyclesListResponse> {
  const params = cursor ? `?cursor=${encodeURIComponent(cursor)}` : "";
  return request<CyclesListResponse>(`/teams/${teamId}/cycles${params}`, {
    method: "GET",
    auth: true,
  });
}

// =============================================================================
// Dashboard Endpoints
// =============================================================================

export async function getMemberDashboard(
  teamId: string
): Promise<MemberDashboardResponse> {
  return request<MemberDashboardResponse>(`/teams/${teamId}/dashboards/member`, {
    method: "GET",
    auth: true,
  });
}

// =============================================================================
// Practice Log Endpoints
// =============================================================================

export async function createPracticeLog(
  cycleId: string,
  data: CreatePracticeLogRequest
): Promise<CreatePracticeLogResponse> {
  return request<CreatePracticeLogResponse>(`/cycles/${cycleId}/practice-logs`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export interface ListPracticeLogsFilters {
  me?: boolean;
  section?: string;
}

export async function listPracticeLogs(
  cycleId: string,
  filters?: ListPracticeLogsFilters,
  cursor?: string
): Promise<PracticeLogsListResponse> {
  const params = new URLSearchParams();
  if (filters?.me !== undefined) params.append("me", String(filters.me));
  if (filters?.section) params.append("section", filters.section);
  if (cursor) params.append("cursor", cursor);
  const queryString = params.toString();
  return request<PracticeLogsListResponse>(
    `/cycles/${cycleId}/practice-logs${queryString ? `?${queryString}` : ""}`,
    {
      method: "GET",
      auth: true,
    }
  );
}

// =============================================================================
// Assignment Endpoints
// =============================================================================

export interface ListAssignmentsFilters {
  scope?: string;
  section?: string;
  priority?: string;
  type?: string;
  song_ref?: string;
}

export async function listAssignments(
  cycleId: string,
  filters?: ListAssignmentsFilters,
  cursor?: string
): Promise<AssignmentsListResponse> {
  const params = new URLSearchParams();
  if (filters?.scope) params.append("scope", filters.scope);
  if (filters?.section) params.append("section", filters.section);
  if (filters?.priority) params.append("priority", filters.priority);
  if (filters?.type) params.append("type", filters.type);
  if (filters?.song_ref) params.append("song_ref", filters.song_ref);
  if (cursor) params.append("cursor", cursor);
  const queryString = params.toString();
  return request<AssignmentsListResponse>(
    `/cycles/${cycleId}/assignments${queryString ? `?${queryString}` : ""}`,
    {
      method: "GET",
      auth: true,
    }
  );
}

// =============================================================================
// Leader Dashboard Endpoints
// =============================================================================

export async function getLeaderDashboard(
  teamId: string
): Promise<LeaderDashboardResponse> {
  return request<LeaderDashboardResponse>(
    `/teams/${teamId}/dashboards/leader`,
    {
      method: "GET",
      auth: true,
    }
  );
}

export async function getComplianceInsights(
  teamId: string
): Promise<ComplianceInsightsResponse> {
  return request<ComplianceInsightsResponse>(
    `/teams/${teamId}/dashboards/leader/compliance-insights`,
    {
      method: "GET",
      auth: true,
    }
  );
}

// =============================================================================
// Team Management Endpoints
// =============================================================================

export async function updateTeam(
  teamId: string,
  data: UpdateTeamRequest
): Promise<UpdateTeamResponse> {
  return request<UpdateTeamResponse>(`/teams/${teamId}`, {
    method: "PATCH",
    body: data,
    auth: true,
  });
}

export async function getTeamMembers(
  teamId: string,
  cursor?: string
): Promise<MembersListResponse> {
  const params = cursor ? `?cursor=${encodeURIComponent(cursor)}` : "";
  return request<MembersListResponse>(`/teams/${teamId}/members${params}`, {
    method: "GET",
    auth: true,
  });
}

export async function updateMember(
  teamId: string,
  userId: string,
  data: UpdateMemberRequest
): Promise<UpdateMemberResponse> {
  return request<UpdateMemberResponse>(`/teams/${teamId}/members/${userId}`, {
    method: "PATCH",
    body: data,
    auth: true,
  });
}

export async function removeMember(
  teamId: string,
  userId: string
): Promise<RevokeInviteResponse> {
  return request<RevokeInviteResponse>(`/teams/${teamId}/members/${userId}`, {
    method: "DELETE",
    auth: true,
  });
}

// =============================================================================
// Team Invites Endpoints
// =============================================================================

export async function createInvite(
  teamId: string,
  data: CreateInviteRequest = {}
): Promise<CreateInviteResponse> {
  return request<CreateInviteResponse>(`/teams/${teamId}/invites`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export async function listInvites(
  teamId: string,
  includeUsed?: boolean,
  cursor?: string
): Promise<InvitesListResponse> {
  const params = new URLSearchParams();
  if (includeUsed) params.append("include_used", "true");
  if (cursor) params.append("cursor", cursor);
  const queryString = params.toString();
  return request<InvitesListResponse>(
    `/teams/${teamId}/invites${queryString ? `?${queryString}` : ""}`,
    {
      method: "GET",
      auth: true,
    }
  );
}

export async function revokeInvite(
  inviteId: string
): Promise<RevokeInviteResponse> {
  return request<RevokeInviteResponse>(`/invites/${inviteId}`, {
    method: "DELETE",
    auth: true,
  });
}

// =============================================================================
// Ticket Endpoints
// =============================================================================

export async function createTicket(
  cycleId: string,
  data: CreateTicketRequest
): Promise<CreateTicketResponse> {
  return request<CreateTicketResponse>(`/cycles/${cycleId}/tickets`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export interface ListTicketsFilters {
  status?: string;
  priority?: string;
  category?: string;
  visibility?: string;
  section?: string;
  song_ref?: string;
}

export async function listTickets(
  cycleId: string,
  filters?: ListTicketsFilters,
  cursor?: string
): Promise<TicketsListResponse> {
  const params = new URLSearchParams();
  if (filters?.status) params.append("status", filters.status);
  if (filters?.priority) params.append("priority", filters.priority);
  if (filters?.category) params.append("category", filters.category);
  if (filters?.visibility) params.append("visibility", filters.visibility);
  if (filters?.section) params.append("section", filters.section);
  if (filters?.song_ref) params.append("song_ref", filters.song_ref);
  if (cursor) params.append("cursor", cursor);
  const queryString = params.toString();
  return request<TicketsListResponse>(
    `/cycles/${cycleId}/tickets${queryString ? `?${queryString}` : ""}`,
    {
      method: "GET",
      auth: true,
    }
  );
}

export async function claimTicket(
  ticketId: string
): Promise<ClaimTicketResponse> {
  return request<ClaimTicketResponse>(`/tickets/${ticketId}/claim`, {
    method: "POST",
    auth: true,
  });
}

export async function updateTicket(
  ticketId: string,
  data: UpdateTicketRequest
): Promise<UpdateTicketResponse> {
  return request<UpdateTicketResponse>(`/tickets/${ticketId}`, {
    method: "PATCH",
    body: data,
    auth: true,
  });
}

export async function transitionTicket(
  ticketId: string,
  data: TransitionTicketRequest
): Promise<TransitionTicketResponse> {
  return request<TransitionTicketResponse>(`/tickets/${ticketId}/transition`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export async function verifyTicket(
  ticketId: string,
  data: VerifyTicketRequest = {}
): Promise<VerifyTicketResponse> {
  return request<VerifyTicketResponse>(`/tickets/${ticketId}/verify`, {
    method: "POST",
    body: data,
    auth: true,
  });
}

export async function getTicketActivity(
  ticketId: string
): Promise<TicketActivitiesResponse> {
  return request<TicketActivitiesResponse>(`/tickets/${ticketId}/activity`, {
    method: "GET",
    auth: true,
  });
}

// =============================================================================
// Export all for easy importing
// =============================================================================

export const api = {
  // Auth
  register,
  login,
  refreshToken,
  getMe,
  // Invites
  getInvitePreview,
  acceptInvite,
  createInvite,
  listInvites,
  revokeInvite,
  // Cycles
  getActiveCycle,
  createCycle,
  listCycles,
  // Dashboards
  getMemberDashboard,
  getLeaderDashboard,
  getComplianceInsights,
  // Practice Logs
  createPracticeLog,
  listPracticeLogs,
  // Assignments
  listAssignments,
  // Team Management
  updateTeam,
  getTeamMembers,
  updateMember,
  removeMember,
  // Tickets
  createTicket,
  listTickets,
  claimTicket,
  updateTicket,
  transitionTicket,
  verifyTicket,
  getTicketActivity,
  // Token management
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
};
