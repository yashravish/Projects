/**
 * API Types - Mirrors backend schemas
 *
 * CONTRACT LOCK PROTOCOL: These types must match the backend OpenAPI spec.
 * If types don't match, update backend first, then regenerate these types.
 */

// =============================================================================
// Enums (Shared Domain Enums)
// =============================================================================

export type Role = "MEMBER" | "SECTION_LEADER" | "ADMIN";

export type AssignmentType =
  | "SONG_WORK"
  | "TECHNIQUE"
  | "MEMORIZATION"
  | "LISTENING";

export type Priority = "LOW" | "MEDIUM" | "BLOCKING";

export type AssignmentScope = "TEAM" | "SECTION";

export type TicketCategory =
  | "PITCH"
  | "RHYTHM"
  | "MEMORY"
  | "BLEND"
  | "TECHNIQUE"
  | "OTHER";

export type TicketVisibility = "PRIVATE" | "SECTION" | "TEAM";

export type TicketStatus =
  | "OPEN"
  | "IN_PROGRESS"
  | "BLOCKED"
  | "RESOLVED"
  | "VERIFIED";

export type TicketActivityType =
  | "CREATED"
  | "COMMENT"
  | "STATUS_CHANGE"
  | "VERIFIED"
  | "CLAIMED"
  | "REASSIGNED";

// =============================================================================
// Error Response
// =============================================================================

export type ErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "VALIDATION_ERROR"
  | "CONFLICT"
  | "RATE_LIMITED"
  | "INTERNAL";

export interface ApiError {
  error: {
    code: ErrorCode;
    message: string;
    field: string | null;
  };
}

// =============================================================================
// Auth Types
// =============================================================================

export interface User {
  id: string;
  email: string;
  name: string;
}

export interface TeamMembership {
  team_id: string;
  role: Role;
  section: string | null;
}

export interface RegisterRequest {
  email: string;
  name: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  user: User;
}

export interface RefreshResponse {
  access_token: string;
}

export interface MeResponse {
  user: User;
  primary_team: TeamMembership | null;
}

// =============================================================================
// Invite Types
// =============================================================================

export interface InvitePreviewResponse {
  team_name: string;
  email: string | null;
  role: Role;
  section: string | null;
  expired: boolean;
}

export interface AcceptInviteRequest {
  name?: string;
  email?: string;
  password?: string;
}

export interface MembershipResponse {
  id: string;
  team_id: string;
  user_id: string;
  role: Role;
  section: string | null;
  created_at: string;
  updated_at: string;
}

export interface AcceptInviteResponse {
  membership: MembershipResponse;
  access_token: string | null;
  refresh_token: string | null;
}

// =============================================================================
// Cycle Types
// =============================================================================

export interface CycleResponse {
  id: string;
  team_id: string;
  name: string;
  date: string;
  created_at: string;
}

export interface ActiveCycleResponse {
  cycle: CycleResponse | null;
}

// =============================================================================
// Pagination Types
// =============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  next_cursor: string | null;
}

// =============================================================================
// Dashboard Types
// =============================================================================

export interface CycleInfo {
  id: string;
  date: string;
  label: string;
}

export interface AssignmentSummary {
  id: string;
  title: string;
  priority: Priority;
  type: AssignmentType;
  scope: AssignmentScope;
  section: string | null;
  due_at: string | null;
}

export interface TicketDueSoon {
  id: string;
  title: string;
  priority: Priority;
  status: TicketStatus;
  due_at: string | null;
  visibility: TicketVisibility;
}

export interface QuickLogDefaults {
  duration_min_default: number;
}

export interface WeeklySummary {
  practice_days: number;
  streak_days: number;
  total_sessions: number;
}

export interface ProgressSummary {
  tickets_resolved_this_cycle: number;
  tickets_verified_this_cycle: number;
}

export interface MemberDashboardResponse {
  cycle: CycleInfo | null;
  countdown_days: number | null;
  assignments: AssignmentSummary[];
  tickets_due_soon: TicketDueSoon[];
  quick_log_defaults: QuickLogDefaults;
  weekly_summary: WeeklySummary;
  progress: ProgressSummary;
}

// =============================================================================
// Practice Log Types
// =============================================================================

export interface CreatePracticeLogRequest {
  occurred_at?: string;
  duration_min: number;
  notes?: string;
  rating_1_5?: number;
  blocked_flag?: boolean;
  assignment_ids?: string[];
}

export interface PracticeLogAssignment {
  id: string;
  title: string;
  type: string;
}

export interface PracticeLogResponse {
  id: string;
  user_id: string;
  team_id: string;
  cycle_id: string | null;
  duration_minutes: number;
  rating_1_5: number | null;
  blocked_flag: boolean;
  notes: string | null;
  occurred_at: string;
  created_at: string;
  assignments: PracticeLogAssignment[];
}

export interface SuggestedTicket {
  title_suggestion: string;
  due_date: string;
  visibility_default: TicketVisibility;
  priority_default: Priority;
  category_default: TicketCategory;
}

export interface CreatePracticeLogResponse {
  practice_log: PracticeLogResponse;
  suggested_ticket: SuggestedTicket | null;
}

export interface PracticeLogsListResponse {
  items: PracticeLogResponse[];
  next_cursor: string | null;
}

// =============================================================================
// Assignment Types (Full List)
// =============================================================================

export interface AssignmentResponse {
  id: string;
  cycle_id: string;
  created_by: string | null;
  type: AssignmentType;
  scope: AssignmentScope;
  priority: Priority;
  section: string | null;
  title: string;
  song_ref: string | null;
  notes: string | null;
  due_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface AssignmentsListResponse {
  items: AssignmentResponse[];
  next_cursor: string | null;
}

// =============================================================================
// Team Management Types
// =============================================================================

export interface TeamResponse {
  id: string;
  name: string;
  created_at: string;
}

export interface UpdateTeamRequest {
  name: string;
}

export interface UpdateTeamResponse {
  team: TeamResponse;
}

export interface MemberResponse {
  id: string;
  user_id: string;
  email: string;
  display_name: string;
  role: Role;
  section: string | null;
  created_at: string;
}

export interface MembersListResponse {
  items: MemberResponse[];
  next_cursor: string | null;
}

export interface UpdateMemberRequest {
  role?: Role;
  section?: string | null;
  primary_team?: boolean;
}

export interface UpdateMemberResponse {
  membership: MembershipResponse;
}

export interface CreateInviteRequest {
  email?: string;
  role?: Role;
  section?: string;
  expires_in_hours?: number;
}

export interface CreateInviteResponse {
  invite_link: string;
}

export interface InviteResponse {
  id: string;
  email: string | null;
  role: Role;
  section: string | null;
  expires_at: string;
  used_at: string | null;
  created_at: string;
}

export interface InvitesListResponse {
  items: InviteResponse[];
  next_cursor: string | null;
}

export interface RevokeInviteResponse {
  message: string;
}

// =============================================================================
// Leader Dashboard Types
// =============================================================================

export type DueBucket =
  | "overdue"
  | "due_today"
  | "due_this_week"
  | "future"
  | "no_due_date";

export interface MemberPracticeDays {
  member_id: string;
  name: string;
  section: string | null;
  days_logged_7d: number;
}

export interface ComplianceSummary {
  logged_last_7_days_pct: number;
  practice_days_by_member: MemberPracticeDays[];
  total_practice_minutes_7d: number;
}

export interface SectionRisk {
  section: string;
  blocking_due: number;
  blocked: number;
  resolved_not_verified: number;
}

export interface SongRisk {
  song_ref: string;
  blocking_due: number;
  blocked: number;
  resolved_not_verified: number;
}

export interface RiskSummary {
  blocking_due_count: number;
  blocked_count: number;
  resolved_not_verified_count: number;
  by_section: SectionRisk[];
  by_song: SongRisk[];
}

export interface PrivateTicketAggregate {
  section: string | null;
  category: TicketCategory;
  status: TicketStatus;
  priority: Priority;
  song_ref: string | null;
  due_bucket: DueBucket;
  count: number;
}

export interface MemberDrilldown {
  member_id: string;
  name: string;
  section: string | null;
  open_ticket_count: number;
  blocked_count: number;
}

export interface TicketVisible {
  id: string;
  title: string;
  priority: Priority;
  status: TicketStatus;
  visibility: TicketVisibility;
  section: string | null;
  due_at: string | null;
}

export interface DrilldownData {
  members: MemberDrilldown[];
  tickets_visible: TicketVisible[];
}

export interface LeaderDashboardResponse {
  cycle: CycleInfo | null;
  compliance: ComplianceSummary;
  risk_summary: RiskSummary;
  private_ticket_aggregates: PrivateTicketAggregate[];
  drilldown: DrilldownData;
}

export type SummarySource = "openai" | "fallback";

export interface ComplianceSectionDatum {
  section: string;
  member_count: number;
  total_practice_days_7d: number;
  avg_practice_days_7d: number;
}

export interface ComplianceInsightsResponse {
  sections: ComplianceSectionDatum[];
  summary: string;
  summary_source: SummarySource;
  window_days: number;
}

// =============================================================================
// Ticket Types
// =============================================================================

export interface TicketResponse {
  id: string;
  team_id: string;
  cycle_id: string | null;
  owner_id: string | null;
  created_by: string;
  claimed_by: string | null;
  claimable: boolean;
  category: TicketCategory;
  priority: Priority;
  status: TicketStatus;
  visibility: TicketVisibility;
  section: string | null;
  title: string;
  description: string | null;
  song_ref: string | null;
  due_at: string | null;
  resolved_at: string | null;
  resolved_note: string | null;
  verified_by: string | null;
  verified_at: string | null;
  verified_note: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateTicketRequest {
  category: TicketCategory;
  priority: Priority;
  visibility: TicketVisibility;
  title: string;
  description?: string;
  song_ref?: string;
  section?: string;
  owner_id?: string;
  claimable?: boolean;
}

export interface CreateTicketResponse {
  ticket: TicketResponse;
}

export interface TicketsListResponse {
  items: TicketResponse[];
  next_cursor: string | null;
}

export interface TransitionTicketRequest {
  to_status: TicketStatus;
  content?: string;
}

export interface TransitionTicketResponse {
  ticket: TicketResponse;
}

export interface VerifyTicketRequest {
  content?: string;
}

export interface VerifyTicketResponse {
  ticket: TicketResponse;
}

export interface ClaimTicketResponse {
  ticket: TicketResponse;
}

export interface UpdateTicketRequest {
  title?: string;
  description?: string;
  category?: TicketCategory;
  song_ref?: string;
  priority?: Priority;
  status?: TicketStatus;
  visibility?: TicketVisibility;
  section?: string;
}

export interface UpdateTicketResponse {
  ticket: TicketResponse;
}

export interface TicketActivityResponse {
  id: string;
  ticket_id: string;
  user_id: string;
  type: TicketActivityType;
  old_status: TicketStatus | null;
  new_status: TicketStatus | null;
  content: string | null;
  created_at: string;
}

export interface TicketActivitiesResponse {
  items: TicketActivityResponse[];
}
