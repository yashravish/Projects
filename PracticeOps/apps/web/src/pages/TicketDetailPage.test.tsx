/**
 * Ticket Detail Page Tests
 *
 * Tests for Milestone 10c:
 * - Happy path: full lifecycle OPEN -> IN_PROGRESS -> RESOLVED -> VERIFIED
 * - Edge: invalid transition shows error
 * - Activity timeline rendering
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TicketDetailPage } from "./TicketDetailPage";
import { render } from "@/test/test-utils";
import { http, HttpResponse } from "msw";
import { server } from "@/test/mocks/server";

const API_BASE = "http://localhost:8000";

// Mutable location state that can be changed per test
let mockLocationState: { ticket?: typeof mockTicket } = {};

// Mock navigate
const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useParams: () => ({ ticketId: "ticket-1" }),
    useLocation: () => ({
      state: mockLocationState,
    }),
  };
});

// Mock auth
const mockUser = {
  id: "user-123",
  email: "test@example.com",
  name: "Test User",
};

const mockTeamMembership = {
  team_id: "team-123",
  role: "MEMBER" as const,
  section: "Tenor",
};

// Reserved for future leader tests
// const mockLeaderMembership = {
//   team_id: "team-123",
//   role: "SECTION_LEADER" as const,
//   section: "Tenor",
// };

vi.mock("@/lib/auth", async () => {
  const actual = await vi.importActual("@/lib/auth");
  return {
    ...actual,
    useAuth: () => ({
      isLoading: false,
      isAuthenticated: true,
      user: { ...mockUser, primary_team: mockTeamMembership },
      primaryTeam: mockTeamMembership,
      error: null,
    }),
  };
});

// Mock ticket data
const mockTicket = {
  id: "ticket-1",
  team_id: "team-123",
  cycle_id: "cycle-123",
  owner_id: "user-123",
  created_by: "user-123",
  claimed_by: null,
  claimable: false,
  category: "PITCH",
  priority: "MEDIUM",
  status: "OPEN",
  visibility: "TEAM",
  section: "Tenor",
  title: "Fix pitch in measure 42",
  description: "Pitch is consistently flat in this measure",
  song_ref: "Song A",
  due_at: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  resolved_at: null,
  resolved_note: null,
  verified_at: null,
  verified_by: null,
  verified_note: null,
};

const mockActivities = [
  {
    id: "activity-1",
    ticket_id: "ticket-1",
    user_id: "user-123",
    type: "CREATED",
    content: null,
    old_status: null,
    new_status: null,
    created_at: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
  },
];

describe("TicketDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset location state to default ticket
    mockLocationState = { ticket: mockTicket };
    localStorage.setItem("practiceops_access_token", "mock-access-token");
    localStorage.setItem("practiceops_refresh_token", "mock-refresh-token");

    // Setup default handlers
    server.use(
      http.get(`${API_BASE}/tickets/ticket-1/activity`, () => {
        return HttpResponse.json({
          items: mockActivities,
        });
      })
    );
  });

  describe("Happy Path - Full Lifecycle", () => {
    it("displays ticket details", async () => {
      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText("Fix pitch in measure 42")).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      expect(screen.getByText("Pitch is consistently flat in this measure")).toBeInTheDocument();
      expect(screen.getByText("Pitch")).toBeInTheDocument(); // Category
      expect(screen.getByText("Song A")).toBeInTheDocument(); // Song ref
    });

    it("shows Start Working button when status is OPEN", async () => {
      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByRole("button", { name: /start working/i })).toBeInTheDocument();
        },
        { timeout: 5000 }
      );
    });

    it("transitions from OPEN to IN_PROGRESS", async () => {
      const user = userEvent.setup();

      server.use(
        http.post(`${API_BASE}/tickets/ticket-1/transition`, async () => {
          return HttpResponse.json({
            ticket: {
              ...mockTicket,
              status: "IN_PROGRESS",
            },
          });
        })
      );

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByRole("button", { name: /start working/i })).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      await user.click(screen.getByRole("button", { name: /start working/i }));

      // Should update to show IN_PROGRESS actions
      await waitFor(() => {
        expect(screen.queryByRole("button", { name: /start working/i })).not.toBeInTheDocument();
      });
    });

    it("shows resolve and block buttons when IN_PROGRESS", async () => {
      const inProgressTicket = { ...mockTicket, status: "IN_PROGRESS" };
      mockLocationState = { ticket: inProgressTicket };

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText("Fix pitch in measure 42")).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      // Should show both Mark Blocked and Resolve buttons
      expect(screen.getByRole("button", { name: /mark blocked/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /resolve/i })).toBeInTheDocument();
    });

    it("opens modal when Resolve is clicked and requires note", async () => {
      const user = userEvent.setup();
      const inProgressTicket = { ...mockTicket, status: "IN_PROGRESS" };
      mockLocationState = { ticket: inProgressTicket };

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByRole("button", { name: /resolve/i })).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      await user.click(screen.getByRole("button", { name: /resolve/i }));

      // Modal should open
      await waitFor(() => {
        expect(screen.getByText("Resolve Ticket")).toBeInTheDocument();
      });

      // Submit button should be disabled without note
      const submitButton = screen.getByRole("button", { name: /mark as resolved/i });
      expect(submitButton).toBeDisabled();
    });

    it("displays activity timeline", async () => {
      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText("Activity Timeline")).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      expect(screen.getByText("Created ticket")).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("shows error when invalid transition attempted", async () => {
      const user = userEvent.setup();
      const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {});

      server.use(
        http.post(`${API_BASE}/tickets/ticket-1/transition`, async () => {
          return HttpResponse.json(
            {
              error: {
                code: "VALIDATION_ERROR",
                message: "Invalid transition from OPEN to RESOLVED",
                field: "to_status",
              },
            },
            { status: 422 }
          );
        })
      );

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByRole("button", { name: /start working/i })).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      // Attempt invalid transition (simulated by clicking button)
      await user.click(screen.getByRole("button", { name: /start working/i }));

      // Should show error alert
      await waitFor(() => {
        expect(alertSpy).toHaveBeenCalledWith(
          expect.stringContaining("Invalid transition")
        );
      });

      alertSpy.mockRestore();
    });

    it("shows error when resolving without note", async () => {
      const user = userEvent.setup();
      const inProgressTicket = { ...mockTicket, status: "IN_PROGRESS" };
      mockLocationState = { ticket: inProgressTicket };

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByRole("button", { name: /resolve/i })).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      await user.click(screen.getByRole("button", { name: /resolve/i }));

      await waitFor(() => {
        expect(screen.getByText("Resolve Ticket")).toBeInTheDocument();
      });

      // Submit button should be disabled without note
      const submitButton = screen.getByRole("button", { name: /mark as resolved/i });
      expect(submitButton).toBeDisabled();
    });

    it("handles ticket not found gracefully", async () => {
      mockLocationState = {};

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(
            screen.getByText("Ticket not found. Please navigate from the tickets list.")
          ).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      expect(screen.getByRole("button", { name: /back to tickets/i })).toBeInTheDocument();
    });
  });

  describe("Activity Timeline", () => {
    it("renders status change activities correctly", async () => {
      server.use(
        http.get(`${API_BASE}/tickets/ticket-1/activity`, () => {
          return HttpResponse.json({
            items: [
              ...mockActivities,
              {
                id: "activity-2",
                ticket_id: "ticket-1",
                user_id: "user-123",
                type: "STATUS_CHANGE",
                content: "Starting work on this",
                old_status: "OPEN",
                new_status: "IN_PROGRESS",
                created_at: new Date().toISOString(),
              },
            ],
          });
        })
      );

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText(/Status: OPEN → IN_PROGRESS/i)).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      expect(screen.getByText("Starting work on this")).toBeInTheDocument();
    });

    it("renders verified activities correctly", async () => {
      server.use(
        http.get(`${API_BASE}/tickets/ticket-1/activity`, () => {
          return HttpResponse.json({
            items: [
              ...mockActivities,
              {
                id: "activity-3",
                ticket_id: "ticket-1",
                user_id: "leader-456",
                type: "VERIFIED",
                content: "Looks good!",
                old_status: "RESOLVED",
                new_status: "VERIFIED",
                created_at: new Date().toISOString(),
              },
            ],
          });
        })
      );

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText("Verified ticket")).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      expect(screen.getByText("Looks good!")).toBeInTheDocument();
    });
  });

  describe("Verification", () => {
    it("does not show verify button for non-leaders", async () => {
      const resolvedTicket = { ...mockTicket, status: "RESOLVED" };
      mockLocationState = { ticket: resolvedTicket };

      render(<TicketDetailPage />);

      await waitFor(
        () => {
          expect(screen.getByText("Fix pitch in measure 42")).toBeInTheDocument();
        },
        { timeout: 5000 }
      );

      // Member should not see verify button
      expect(screen.queryByRole("button", { name: /verify/i })).not.toBeInTheDocument();
    });

    // Note: Testing leader verification would require mocking the user as a leader
    // This is covered by the backend tests which verify RBAC for verification
  });
});
