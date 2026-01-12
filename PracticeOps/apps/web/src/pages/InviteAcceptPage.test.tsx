/**
 * Invite Accept Page Tests
 */

import { describe, it, expect, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { Routes, Route } from "react-router-dom";
import { renderWithRouter } from "@/test/test-utils";
import { InviteAcceptPage } from "./InviteAcceptPage";

// Mock react-router-dom useNavigate
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

describe("InviteAcceptPage", () => {
  it("shows loading state initially", () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/valid-token"] }
    );

    // Should show skeleton loading
    expect(document.querySelector(".animate-pulse")).toBeInTheDocument();
  });

  it("shows invite details after loading", async () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/valid-token"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/test team/i)).toBeInTheDocument();
      expect(screen.getByText(/member/i)).toBeInTheDocument();
    });
  });

  it("shows expired message for expired invite", async () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/expired-token"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/expired/i)).toBeInTheDocument();
    });
  });

  it("shows error for invalid invite", async () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/invalid-token"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/invalid/i)).toBeInTheDocument();
    });
  });

  it("shows registration form for unauthenticated users", async () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/valid-token"] }
    );

    await waitFor(() => {
      expect(screen.getByLabelText(/display name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });
  });

  it("pre-fills email from invite", async () => {
    renderWithRouter(
      <Routes>
        <Route path="/invites/:token" element={<InviteAcceptPage />} />
      </Routes>,
      { initialEntries: ["/invites/email-token"] }
    );

    await waitFor(() => {
      const emailInput = screen.getByLabelText(/email/i) as HTMLInputElement;
      expect(emailInput.value).toBe("preset@example.com");
    });
  });
});

