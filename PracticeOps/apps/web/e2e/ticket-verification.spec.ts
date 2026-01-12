/**
 * E2E Test: Ticket Verification Workflow
 *
 * Tests the complete flow:
 * 1. Admin registers → creates team
 * 2. Admin creates cycle and invites leader + member
 * 3. Member: accepts invite, creates ticket, transitions OPEN → IN_PROGRESS → RESOLVED
 * 4. Leader: logs in, opens /leader dashboard, verifies ticket
 *
 * Prerequisites:
 * - Backend API must be running on http://localhost:8000
 * - Frontend dev server must be running on http://localhost:5173
 * - Database should be clean for test isolation
 */

import { test, expect, Page } from "@playwright/test";

const API_BASE = "http://localhost:8000";

// Helper to generate unique emails for test isolation
const uniqueId = () => Math.random().toString(36).substring(7);

interface TestUser {
  email: string;
  password: string;
  name: string;
  accessToken?: string;
  refreshToken?: string;
  userId?: string;
}

// Helper to register a user directly via API
async function registerUser(userData: { email: string; name: string; password: string }): Promise<{
  accessToken: string;
  refreshToken: string;
  userId: string;
}> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: userData.email,
        name: userData.name,
        password: userData.password,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        accessToken: data.access_token,
        refreshToken: data.refresh_token,
        userId: data.user.id,
      };
    }

    if (response.status === 429 && attempt < 2) {
      const retryAfter = Number(response.headers.get("Retry-After") ?? 0);
      const waitMs = retryAfter > 0 ? retryAfter * 1000 : (attempt + 1) * 3000;
      await new Promise((resolve) => setTimeout(resolve, waitMs));
      continue;
    }

    const error = await response.json();
    throw new Error(`Failed to register: ${JSON.stringify(error)}`);
  }

  throw new Error("Failed to register after retries");
}

// Helper to create a team
async function createTeam(accessToken: string, name: string): Promise<string> {
  const response = await fetch(`${API_BASE}/teams`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to create team: ${JSON.stringify(error)}`);
  }

  const data = await response.json();
  return data.team.id;
}

// Helper to create an invite
async function createInvite(
  teamId: string,
  accessToken: string,
  options: { email?: string; role?: string; section?: string }
): Promise<string> {
  const response = await fetch(`${API_BASE}/teams/${teamId}/invites`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      email: options.email,
      role: options.role || "MEMBER",
      section: options.section,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to create invite: ${JSON.stringify(error)}`);
  }

  const data = await response.json();
  // Extract token from invite_link
  const token = data.invite_link.split("/invites/")[1];
  return token;
}

// Helper to create a cycle
async function createCycle(
  teamId: string,
  accessToken: string,
  date: Date
): Promise<string> {
  const dateString = date.toISOString().split("T")[0];
  const response = await fetch(`${API_BASE}/teams/${teamId}/cycles`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      name: "Test Rehearsal",
      date: dateString,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to create cycle: ${JSON.stringify(error)}`);
  }

  const data = await response.json();
  return data.cycle.id;
}

// Helper to create a ticket
async function createTicket(
  cycleId: string,
  accessToken: string,
  ticketData: {
    title: string;
    category: string;
    priority: string;
    visibility: string;
    section?: string;
  }
): Promise<string> {
  const response = await fetch(`${API_BASE}/cycles/${cycleId}/tickets`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(ticketData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to create ticket: ${JSON.stringify(error)}`);
  }

  const data = await response.json();
  return data.ticket.id;
}

// Helper to create a practice log
async function createPracticeLog(
  cycleId: string,
  accessToken: string,
  occurredAt: string
): Promise<void> {
  const response = await fetch(`${API_BASE}/cycles/${cycleId}/practice-logs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      duration_min: 30,
      assignment_ids: [],
      blocked_flag: false,
      occurred_at: occurredAt,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to create practice log: ${JSON.stringify(error)}`);
  }
}

// Helper to transition a ticket
async function transitionTicket(
  ticketId: string,
  accessToken: string,
  toStatus: string,
  content?: string
): Promise<void> {
  const response = await fetch(`${API_BASE}/tickets/${ticketId}/transition`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      to_status: toStatus,
      content: content,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Failed to transition ticket: ${JSON.stringify(error)}`);
  }
}

// Helper to log in via UI
async function loginViaUI(
  page: Page,
  email: string,
  password: string,
  tokens?: { accessToken: string; refreshToken: string }
): Promise<void> {
  if (tokens?.accessToken && tokens?.refreshToken) {
    await page.addInitScript(
      ({ accessToken, refreshToken }) => {
        localStorage.setItem("practiceops_access_token", accessToken);
        localStorage.setItem("practiceops_refresh_token", refreshToken);
      },
      tokens
    );
    await page.goto("/dashboard");
    await expect(page).toHaveURL("/dashboard", { timeout: 10000 });
    return;
  }

  await page.goto("/login");
  await page.getByLabel(/email/i).fill(email);
  await page.getByLabel(/password/i).fill(password);
  await page.getByRole("button", { name: /sign in/i }).click();

  // Wait for redirect to dashboard
  await expect(page).toHaveURL("/dashboard", { timeout: 10000 });
}

test.describe("Ticket Verification Workflow", () => {
  test.describe.configure({ mode: "serial" });
  let admin: TestUser;
  let leader: TestUser;
  let member: TestUser;
  let teamId: string;
  let cycleId: string;
  let ticketId: string;

  test.beforeAll(async () => {
    const testId = uniqueId();

    // 1. Create admin and team
    admin = {
      email: `admin-${testId}@test.com`,
      password: "password123",
      name: "Test Admin",
    };
    const adminAuth = await registerUser(admin);
    admin.accessToken = adminAuth.accessToken;
    admin.refreshToken = adminAuth.refreshToken;
    admin.userId = adminAuth.userId;

    teamId = await createTeam(admin.accessToken!, "E2E Test Team");

    // Create a cycle for next week
    const nextWeek = new Date();
    nextWeek.setDate(nextWeek.getDate() + 7);
    cycleId = await createCycle(teamId, admin.accessToken!, nextWeek);

    // 2. Create leader invite and accept
    leader = {
      email: `leader-${testId}@test.com`,
      password: "password123",
      name: "Test Leader",
    };
    const leaderInviteToken = await createInvite(teamId, admin.accessToken!, {
      email: leader.email,
      role: "SECTION_LEADER",
      section: "Soprano",
    });

    // Accept leader invite (creates account)
    const leaderAcceptResponse = await fetch(`${API_BASE}/invites/${leaderInviteToken}/accept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: leader.name,
        email: leader.email,
        password: leader.password,
      }),
    });
    const leaderAcceptData = await leaderAcceptResponse.json();
    leader.accessToken = leaderAcceptData.access_token;
    leader.refreshToken = leaderAcceptData.refresh_token;

    // 3. Create member invite and accept
    member = {
      email: `member-${testId}@test.com`,
      password: "password123",
      name: "Test Member",
    };
    const memberInviteToken = await createInvite(teamId, admin.accessToken!, {
      email: member.email,
      role: "MEMBER",
      section: "Soprano",
    });

    // Accept member invite (creates account)
    const memberAcceptResponse = await fetch(`${API_BASE}/invites/${memberInviteToken}/accept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: member.name,
        email: member.email,
        password: member.password,
      }),
    });
    const memberAcceptData = await memberAcceptResponse.json();
    member.accessToken = memberAcceptData.access_token;
    member.refreshToken = memberAcceptData.refresh_token;

    // 4. Member creates ticket
    ticketId = await createTicket(cycleId, member.accessToken!, {
      title: "E2E Test Ticket - Pitch Issue",
      category: "PITCH",
      priority: "MEDIUM",
      visibility: "SECTION",
      section: "Soprano",
    });

    // 5. Member transitions ticket: OPEN → IN_PROGRESS
    await transitionTicket(ticketId, member.accessToken!, "IN_PROGRESS");

    // 6. Member transitions ticket: IN_PROGRESS → RESOLVED
    await transitionTicket(ticketId, member.accessToken!, "RESOLVED", "Fixed the pitch issue in measure 23");

    // 7. Add practice logs for compliance insights
    const now = new Date();
    await createPracticeLog(cycleId, leader.accessToken!, now.toISOString());
    const yesterday = new Date();
    yesterday.setDate(now.getDate() - 1);
    await createPracticeLog(cycleId, member.accessToken!, yesterday.toISOString());
  });

  test("member cannot access leader dashboard", async ({ page }) => {
    await loginViaUI(page, member.email, member.password, {
      accessToken: member.accessToken!,
      refreshToken: member.refreshToken!,
    });

    // Try to navigate to leader dashboard
    await page.goto("/leader");

    // Should be redirected to dashboard
    await expect(page).toHaveURL("/dashboard", { timeout: 5000 });
  });

  test("leader can access leader dashboard and see resolved ticket", async ({ page }) => {
    await loginViaUI(page, leader.email, leader.password, {
      accessToken: leader.accessToken!,
      refreshToken: leader.refreshToken!,
    });

    // Navigate to leader dashboard
    await page.goto("/leader");
    await expect(page).toHaveURL("/leader");

    // Should see the Leader Dashboard heading
    await expect(page.getByRole("heading", { name: /leader dashboard/i })).toBeVisible();

    // Navigate to the Tickets tab
    await page.getByRole("tab", { name: /tickets/i }).click();

    // Should see the pending verification section
    await expect(page.getByText(/tickets awaiting verification/i)).toBeVisible();

    // Should see the test ticket
    await expect(page.getByText("E2E Test Ticket - Pitch Issue")).toBeVisible();

    // Should see verify button
    await expect(page.getByRole("button", { name: /verify/i })).toBeVisible();
  });

  test("leader sees compliance insights summary", async ({ page }) => {
    await loginViaUI(page, leader.email, leader.password, {
      accessToken: leader.accessToken!,
      refreshToken: leader.refreshToken!,
    });

    await page.goto("/leader");
    await expect(page).toHaveURL("/leader");

    await expect(page.getByRole("heading", { name: /leader dashboard/i })).toBeVisible();
    await page.getByRole("tab", { name: /compliance/i }).click();

    await expect(page.getByText(/practice compliance by section/i)).toBeVisible({
      timeout: 15000,
    });
    await expect(page.getByText(/summary/i)).toBeVisible();
    await expect(page.locator("text=Soprano").first()).toBeVisible();
  });

  test("leader can verify a resolved ticket", async ({ page }) => {
    await loginViaUI(page, leader.email, leader.password, {
      accessToken: leader.accessToken!,
      refreshToken: leader.refreshToken!,
    });

    // Navigate to leader dashboard tickets tab
    await page.goto("/leader");
    await page.getByRole("tab", { name: /tickets/i }).click();

    // Wait for ticket to appear
    await expect(page.getByText("E2E Test Ticket - Pitch Issue")).toBeVisible();

    // Click verify button
    await page.getByRole("button", { name: /verify/i }).click();

    // Should open verify dialog
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByRole("heading", { name: /verify ticket/i })).toBeVisible();

    // Add a verification note (optional)
    await page.getByPlaceholder(/add any feedback/i).fill("Good work fixing this issue!");

    // Click verify button in dialog
    await page.getByRole("button", { name: /^verify ticket$/i }).click();

    // Should show empty state
    await expect(page.getByText(/no tickets pending verification/i)).toBeVisible();

    // Ticket should disappear from the tickets tab panel
    const ticketsPanel = page.getByRole("tabpanel", { name: /tickets/i });
    await expect(
      ticketsPanel.getByText("E2E Test Ticket - Pitch Issue")
    ).toHaveCount(0);
  });

  test("admin can access team settings", async ({ page }) => {
    await loginViaUI(page, admin.email, admin.password, {
      accessToken: admin.accessToken!,
      refreshToken: admin.refreshToken!,
    });

    // Navigate to settings
    await page.goto("/settings");
    await expect(page).toHaveURL("/settings");

    // Should see Team Settings heading
    await expect(page.getByRole("heading", { name: /team settings/i })).toBeVisible();

    // Should see tabs for Team Info, Members, Invites
    await expect(page.getByRole("tab", { name: /team info/i })).toBeVisible();
    await expect(page.getByRole("tab", { name: /members/i })).toBeVisible();
    await expect(page.getByRole("tab", { name: /invites/i })).toBeVisible();
  });

  test("member cannot access team settings", async ({ page }) => {
    await loginViaUI(page, member.email, member.password, {
      accessToken: member.accessToken!,
      refreshToken: member.refreshToken!,
    });

    // Try to navigate to settings
    await page.goto("/settings");

    // Should be redirected to dashboard
    await expect(page).toHaveURL("/dashboard", { timeout: 5000 });
  });

  test("private ticket aggregates show no identifying information", async ({ page }) => {
    await loginViaUI(page, leader.email, leader.password, {
      accessToken: leader.accessToken!,
      refreshToken: leader.refreshToken!,
    });

    // Navigate to leader dashboard
    await page.goto("/leader");

    // Navigate to Private Aggregates tab
    await page.getByRole("tab", { name: /private aggregates/i }).click();

    // Should see privacy notice
    await expect(page.getByText(/privacy notice/i)).toBeVisible();

    // Should NOT see any ticket IDs (check the table doesn't contain UUID patterns)
    // UUID pattern check - should not contain any UUIDs in the aggregates section
    const uuidPattern = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
    const aggregatesSection = await page
      .getByRole("tabpanel", { name: /private aggregates/i })
      .textContent();

    // The aggregates section should not contain UUIDs (ticket IDs, user IDs)
    const foundUuids = aggregatesSection?.match(uuidPattern) || [];
    expect(foundUuids.length).toBe(0);
  });
});

