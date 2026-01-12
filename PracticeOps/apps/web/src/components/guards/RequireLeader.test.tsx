/**
 * RequireLeader Guard Tests
 */

import { describe, it, expect } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { Routes, Route } from "react-router-dom";
import { renderWithRouter } from "@/test/test-utils";
import { RequireLeader } from "./RequireLeader";
import { server } from "@/test/mocks/server";
import {
  handlers,
  leaderHandlers,
  adminHandlers,
} from "@/test/mocks/handlers";

describe("RequireLeader", () => {
  it("redirects members to dashboard", async () => {
    // Default handlers return MEMBER role
    server.use(...handlers);
    localStorage.setItem("practiceops_access_token", "mock-token");

    renderWithRouter(
      <Routes>
        <Route
          path="/leader"
          element={
            <RequireLeader>
              <div>Leader Content</div>
            </RequireLeader>
          }
        />
        <Route path="/dashboard" element={<div>Dashboard</div>} />
      </Routes>,
      { initialEntries: ["/leader"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
    });
  });

  it("allows section leaders to access", async () => {
    server.use(...leaderHandlers);
    localStorage.setItem("practiceops_access_token", "mock-token");

    renderWithRouter(
      <Routes>
        <Route
          path="/leader"
          element={
            <RequireLeader>
              <div>Leader Content</div>
            </RequireLeader>
          }
        />
        <Route path="/dashboard" element={<div>Dashboard</div>} />
      </Routes>,
      { initialEntries: ["/leader"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/leader content/i)).toBeInTheDocument();
    });
  });

  it("allows admins to access", async () => {
    server.use(...adminHandlers);
    localStorage.setItem("practiceops_access_token", "mock-token");

    renderWithRouter(
      <Routes>
        <Route
          path="/leader"
          element={
            <RequireLeader>
              <div>Leader Content</div>
            </RequireLeader>
          }
        />
        <Route path="/dashboard" element={<div>Dashboard</div>} />
      </Routes>,
      { initialEntries: ["/leader"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/leader content/i)).toBeInTheDocument();
    });
  });
});

