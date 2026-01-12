/**
 * RequireAuth Guard Tests
 */

import { describe, it, expect } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { Routes, Route } from "react-router-dom";
import { renderWithRouter } from "@/test/test-utils";
import { RequireAuth } from "./RequireAuth";
import { server } from "@/test/mocks/server";
import { unauthenticatedHandlers } from "@/test/mocks/handlers";

describe("RequireAuth", () => {
  it("redirects to login when not authenticated", async () => {
    // Override handlers to return 401
    server.use(...unauthenticatedHandlers);

    renderWithRouter(
      <Routes>
        <Route
          path="/protected"
          element={
            <RequireAuth>
              <div>Protected Content</div>
            </RequireAuth>
          }
        />
        <Route path="/login" element={<div>Login Page</div>} />
      </Routes>,
      { initialEntries: ["/protected"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/login page/i)).toBeInTheDocument();
    });
  });

  it("renders children when authenticated", async () => {
    // Default handlers return authenticated user
    localStorage.setItem("practiceops_access_token", "mock-token");

    renderWithRouter(
      <Routes>
        <Route
          path="/protected"
          element={
            <RequireAuth>
              <div>Protected Content</div>
            </RequireAuth>
          }
        />
        <Route path="/login" element={<div>Login Page</div>} />
      </Routes>,
      { initialEntries: ["/protected"] }
    );

    await waitFor(() => {
      expect(screen.getByText(/protected content/i)).toBeInTheDocument();
    });
  });

  it("shows loading state while checking auth", () => {
    localStorage.setItem("practiceops_access_token", "mock-token");

    renderWithRouter(
      <Routes>
        <Route
          path="/protected"
          element={
            <RequireAuth>
              <div>Protected Content</div>
            </RequireAuth>
          }
        />
      </Routes>,
      { initialEntries: ["/protected"] }
    );

    // Should show loading initially
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});

// RequireGuest tests are in a separate file

