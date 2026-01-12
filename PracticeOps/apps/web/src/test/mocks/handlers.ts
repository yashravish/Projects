/**
 * MSW Request Handlers
 *
 * Mock handlers for API endpoints used in tests.
 */

import { http, HttpResponse } from "msw";

const API_BASE = "http://localhost:8000";

// Mock user data
export const mockUser = {
  id: "user-123",
  email: "test@example.com",
  name: "Test User",
};

export const mockTeamMembership = {
  team_id: "team-123",
  role: "MEMBER" as const,
  section: "Tenor",
};

export const mockLeaderMembership = {
  team_id: "team-123",
  role: "SECTION_LEADER" as const,
  section: "Tenor",
};

export const mockAdminMembership = {
  team_id: "team-123",
  role: "ADMIN" as const,
  section: null,
};

// Mock dashboard data
export const mockMemberDashboard = {
  cycle: {
    id: "cycle-123",
    date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
    label: "Week 12 - Spring Concert",
  },
  countdown_days: 3,
  assignments: [
    {
      id: "assignment-1",
      title: "Learn Alto Part mm. 45-60",
      priority: "BLOCKING" as const,
      type: "SONG_WORK" as const,
      scope: "SECTION" as const,
      section: "Tenor",
      due_at: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: "assignment-2",
      title: "Review breath support technique",
      priority: "MEDIUM" as const,
      type: "TECHNIQUE" as const,
      scope: "TEAM" as const,
      section: null,
      due_at: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: "assignment-3",
      title: "Memorize lyrics verse 2",
      priority: "LOW" as const,
      type: "MEMORIZATION" as const,
      scope: "TEAM" as const,
      section: null,
      due_at: null,
    },
  ],
  tickets_due_soon: [
    {
      id: "ticket-1",
      title: "Pitch issue in measure 52",
      priority: "MEDIUM" as const,
      status: "IN_PROGRESS" as const,
      due_at: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
      visibility: "PRIVATE" as const,
    },
  ],
  quick_log_defaults: {
    duration_min_default: 20,
  },
  weekly_summary: {
    practice_days: 4,
    streak_days: 7,
    total_sessions: 6,
  },
  progress: {
    tickets_resolved_this_cycle: 2,
    tickets_verified_this_cycle: 1,
  },
};

export const mockEmptyDashboard = {
  cycle: {
    id: "cycle-123",
    date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
    label: "Week 12 - Spring Concert",
  },
  countdown_days: 3,
  assignments: [],
  tickets_due_soon: [],
  quick_log_defaults: {
    duration_min_default: 20,
  },
  weekly_summary: {
    practice_days: 0,
    streak_days: 0,
    total_sessions: 0,
  },
  progress: {
    tickets_resolved_this_cycle: 0,
    tickets_verified_this_cycle: 0,
  },
};

export const mockNoCycleDashboard = {
  cycle: null,
  countdown_days: null,
  assignments: [],
  tickets_due_soon: [],
  quick_log_defaults: {
    duration_min_default: 20,
  },
  weekly_summary: {
    practice_days: 0,
    streak_days: 0,
    total_sessions: 0,
  },
  progress: {
    tickets_resolved_this_cycle: 0,
    tickets_verified_this_cycle: 0,
  },
};

export const mockComplianceInsights = {
  window_days: 7,
  summary_source: "openai" as const,
  summary: "Soprano leads compliance while Tenor trails, with an overall average of 3.1 days logged.",
  sections: [
    {
      section: "Soprano",
      member_count: 4,
      total_practice_days_7d: 20,
      avg_practice_days_7d: 5,
    },
    {
      section: "Tenor",
      member_count: 3,
      total_practice_days_7d: 6,
      avg_practice_days_7d: 2,
    },
  ],
};

// Auth handlers
export const handlers = [
  // POST /auth/register
  http.post(`${API_BASE}/auth/register`, async ({ request }) => {
    const body = await request.json() as { email: string; name: string; password: string };

    if (body.email === "existing@example.com") {
      return HttpResponse.json(
        {
          error: {
            code: "CONFLICT",
            message: "Email already registered",
            field: "email",
          },
        },
        { status: 409 }
      );
    }

    return HttpResponse.json(
      {
        access_token: "mock-access-token",
        refresh_token: "mock-refresh-token",
        user: {
          id: "new-user-123",
          email: body.email,
          name: body.name,
        },
      },
      { status: 201 }
    );
  }),

  // POST /auth/login
  http.post(`${API_BASE}/auth/login`, async ({ request }) => {
    const body = await request.json() as { email: string; password: string };

    if (body.email === "test@example.com" && body.password === "password123") {
      return HttpResponse.json({
        access_token: "mock-access-token",
        refresh_token: "mock-refresh-token",
        user: mockUser,
      });
    }

    return HttpResponse.json(
      {
        error: {
          code: "UNAUTHORIZED",
          message: "Invalid email or password",
          field: null,
        },
      },
      { status: 401 }
    );
  }),

  // POST /auth/refresh
  http.post(`${API_BASE}/auth/refresh`, async () => {
    return HttpResponse.json({
      access_token: "new-mock-access-token",
    });
  }),

  // GET /me
  http.get(`${API_BASE}/me`, async ({ request }) => {
    const auth = request.headers.get("Authorization");

    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        {
          error: {
            code: "UNAUTHORIZED",
            message: "Not authenticated",
            field: null,
          },
        },
        { status: 401 }
      );
    }

    return HttpResponse.json({
      user: mockUser,
      primary_team: mockTeamMembership,
    });
  }),

  // GET /invites/:token
  http.get(`${API_BASE}/invites/:token`, async ({ params }) => {
    const { token } = params;

    if (token === "expired-token") {
      return HttpResponse.json({
        team_name: "Test Team",
        email: null,
        role: "MEMBER",
        section: null,
        expired: true,
      });
    }

    if (token === "invalid-token") {
      return HttpResponse.json(
        {
          error: {
            code: "NOT_FOUND",
            message: "Invite not found",
            field: null,
          },
        },
        { status: 404 }
      );
    }

    return HttpResponse.json({
      team_name: "Test Team",
      email: token === "email-token" ? "preset@example.com" : null,
      role: "MEMBER",
      section: "Tenor",
      expired: false,
    });
  }),

  // POST /invites/:token/accept
  http.post(`${API_BASE}/invites/:token/accept`, async ({ params, request }) => {
    const { token } = params;
    const auth = request.headers.get("Authorization");

    if (token === "expired-token") {
      return HttpResponse.json(
        {
          error: {
            code: "FORBIDDEN",
            message: "Invite has expired",
            field: null,
          },
        },
        { status: 403 }
      );
    }

    // Logged in user accepting
    if (auth) {
      return HttpResponse.json(
        {
          membership: {
            id: "membership-123",
            team_id: "team-123",
            user_id: mockUser.id,
            role: "MEMBER",
            section: "Tenor",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          access_token: null,
          refresh_token: null,
        },
        { status: 201 }
      );
    }

    // Not logged in - create account
    const body = await request.json() as { name?: string; email?: string; password?: string };

    if (!body.name || !body.email || !body.password) {
      return HttpResponse.json(
        {
          error: {
            code: "CONFLICT",
            message: "Name, email, and password required",
            field: null,
          },
        },
        { status: 409 }
      );
    }

    return HttpResponse.json(
      {
        membership: {
          id: "membership-123",
          team_id: "team-123",
          user_id: "new-user-123",
          role: "MEMBER",
          section: "Tenor",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        access_token: "new-access-token",
        refresh_token: "new-refresh-token",
      },
      { status: 201 }
    );
  }),

  // GET /teams/:teamId/cycles/active
  http.get(`${API_BASE}/teams/:teamId/cycles/active`, async () => {
    return HttpResponse.json({
      cycle: {
        id: "cycle-123",
        team_id: "team-123",
        name: "Week 12",
        date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(), // 3 days from now
        created_at: new Date().toISOString(),
      },
    });
  }),

  // GET /teams/:teamId/dashboards/member
  http.get(`${API_BASE}/teams/:teamId/dashboards/member`, async ({ request }) => {
    const auth = request.headers.get("Authorization");

    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        {
          error: {
            code: "UNAUTHORIZED",
            message: "Not authenticated",
            field: null,
          },
        },
        { status: 401 }
      );
    }

    return HttpResponse.json(mockMemberDashboard);
  }),

  // GET /teams/:teamId/dashboards/leader/compliance-insights
  http.get(`${API_BASE}/teams/:teamId/dashboards/leader/compliance-insights`, async ({ request }) => {
    const auth = request.headers.get("Authorization");

    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        {
          error: {
            code: "UNAUTHORIZED",
            message: "Not authenticated",
            field: null,
          },
        },
        { status: 401 }
      );
    }

    return HttpResponse.json(mockComplianceInsights);
  }),

  // POST /cycles/:cycleId/practice-logs
  http.post(`${API_BASE}/cycles/:cycleId/practice-logs`, async ({ request }) => {
    const auth = request.headers.get("Authorization");

    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        {
          error: {
            code: "UNAUTHORIZED",
            message: "Not authenticated",
            field: null,
          },
        },
        { status: 401 }
      );
    }

    const body = await request.json() as { 
      duration_min: number;
      assignment_ids?: string[];
      blocked_flag?: boolean;
      notes?: string;
      rating_1_5?: number;
    };

    const practiceLog = {
      id: "practice-log-123",
      user_id: mockUser.id,
      team_id: "team-123",
      cycle_id: "cycle-123",
      duration_minutes: body.duration_min,
      rating_1_5: body.rating_1_5 || null,
      blocked_flag: body.blocked_flag || false,
      notes: body.notes || null,
      occurred_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      assignments: (body.assignment_ids || []).map((id) => ({
        id,
        title: `Assignment ${id}`,
        type: "SONG_WORK",
      })),
    };

    // Include suggested ticket if blocked
    const suggestedTicket = body.blocked_flag
      ? {
          title_suggestion: "Practice blocker - needs attention",
          due_date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
          visibility_default: "PRIVATE",
          priority_default: "MEDIUM",
          category_default: "OTHER",
        }
      : null;

    return HttpResponse.json(
      {
        practice_log: practiceLog,
        suggested_ticket: suggestedTicket,
      },
      { status: 201 }
    );
  }),
];

// Handler overrides for specific test scenarios
export const leaderHandlers = [
  http.get(`${API_BASE}/me`, async () => {
    return HttpResponse.json({
      user: mockUser,
      primary_team: mockLeaderMembership,
    });
  }),
];

export const adminHandlers = [
  http.get(`${API_BASE}/me`, async () => {
    return HttpResponse.json({
      user: mockUser,
      primary_team: mockAdminMembership,
    });
  }),
];

export const unauthenticatedHandlers = [
  http.get(`${API_BASE}/me`, async () => {
    return HttpResponse.json(
      {
        error: {
          code: "UNAUTHORIZED",
          message: "Not authenticated",
          field: null,
        },
      },
      { status: 401 }
    );
  }),
];

// Dashboard with no active cycle
export const noCycleHandlers = [
  http.get(`${API_BASE}/teams/:teamId/dashboards/member`, async ({ request }) => {
    const auth = request.headers.get("Authorization");
    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        { error: { code: "UNAUTHORIZED", message: "Not authenticated", field: null } },
        { status: 401 }
      );
    }
    return HttpResponse.json(mockNoCycleDashboard);
  }),
];

// Dashboard with empty data
export const emptyDashboardHandlers = [
  http.get(`${API_BASE}/teams/:teamId/dashboards/member`, async ({ request }) => {
    const auth = request.headers.get("Authorization");
    if (!auth || !auth.startsWith("Bearer ")) {
      return HttpResponse.json(
        { error: { code: "UNAUTHORIZED", message: "Not authenticated", field: null } },
        { status: 401 }
      );
    }
    return HttpResponse.json(mockEmptyDashboard);
  }),
];

// Unauthorized dashboard access (expired token)
export const expiredTokenDashboardHandlers = [
  http.get(`${API_BASE}/teams/:teamId/dashboards/member`, async () => {
    return HttpResponse.json(
      {
        error: {
          code: "UNAUTHORIZED",
          message: "Token expired",
          field: null,
        },
      },
      { status: 401 }
    );
  }),
  // Also fail the refresh
  http.post(`${API_BASE}/auth/refresh`, async () => {
    return HttpResponse.json(
      {
        error: {
          code: "UNAUTHORIZED",
          message: "Refresh token expired",
          field: null,
        },
      },
      { status: 401 }
    );
  }),
];

