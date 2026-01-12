/**
 * App Component Tests
 *
 * Tests for the main App component with proper providers.
 */

import { screen, waitFor } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { renderWithRouter } from "@/test/test-utils";
import App from "./App";
import { server } from "@/test/mocks/server";
import { unauthenticatedHandlers } from "@/test/mocks/handlers";

describe("App", () => {
  it("redirects unauthenticated users to login", async () => {
    // Use unauthenticated handlers
    server.use(...unauthenticatedHandlers);

    renderWithRouter(<App />, { initialEntries: ["/dashboard"] });

    // Should redirect to login
    await waitFor(() => {
      expect(screen.getByText(/welcome back/i)).toBeInTheDocument();
    });
  });

  it("shows login page at /login route", async () => {
    server.use(...unauthenticatedHandlers);

    renderWithRouter(<App />, { initialEntries: ["/login"] });

    await waitFor(() => {
      expect(screen.getByText(/welcome back/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    });
  });

  it("shows register page at /register route", async () => {
    server.use(...unauthenticatedHandlers);

    renderWithRouter(<App />, { initialEntries: ["/register"] });

    await waitFor(() => {
      expect(screen.getByText(/create an account/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/display name/i)).toBeInTheDocument();
    });
  });
});
