/**
 * Empty State Components Tests
 *
 * Tests for empty state components:
 * - Base EmptyState
 * - All preset empty states
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BrowserRouter } from "react-router-dom";
import {
  EmptyState,
  NoAssignmentsEmptyState,
  NoTicketsEmptyState,
  NoPracticeLogsEmptyState,
  NoTeamMembersEmptyState,
  NoFilterResultsEmptyState,
  NoSearchResultsEmptyState,
  NoCycleEmptyState,
  NoInvitesEmptyState,
  NoTicketsToVerifyEmptyState,
} from "./empty-state";

const renderWithRouter = (ui: React.ReactElement) => {
  return render(<BrowserRouter>{ui}</BrowserRouter>);
};

describe("Empty State Components", () => {
  describe("Base EmptyState", () => {
    it("renders title", () => {
      renderWithRouter(<EmptyState title="Test Title" />);
      expect(screen.getByText("Test Title")).toBeInTheDocument();
    });

    it("renders description when provided", () => {
      renderWithRouter(
        <EmptyState title="Title" description="Test description" />
      );
      expect(screen.getByText("Test description")).toBeInTheDocument();
    });

    it("renders icon when provided", () => {
      renderWithRouter(
        <EmptyState
          title="Title"
          icon={<span data-testid="test-icon">Icon</span>}
        />
      );
      expect(screen.getByTestId("test-icon")).toBeInTheDocument();
    });

    it("renders action button when onClick provided", async () => {
      const handleClick = vi.fn();
      renderWithRouter(
        <EmptyState
          title="Title"
          action={{ label: "Click me", onClick: handleClick }}
        />
      );

      const button = screen.getByText("Click me");
      expect(button).toBeInTheDocument();

      await userEvent.click(button);
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it("renders action link when href provided", () => {
      renderWithRouter(
        <EmptyState
          title="Title"
          action={{ label: "Go somewhere", href: "/test-path" }}
        />
      );

      const link = screen.getByRole("link", { name: "Go somewhere" });
      expect(link).toHaveAttribute("href", "/test-path");
    });

    it("uses compact styling when compact prop is true", () => {
      const { container } = renderWithRouter(
        <EmptyState title="Title" compact />
      );
      // Compact uses py-6 instead of py-12
      expect(container.querySelector(".py-6")).toBeInTheDocument();
    });
  });

  describe("NoAssignmentsEmptyState", () => {
    it("renders correct title and description", () => {
      renderWithRouter(<NoAssignmentsEmptyState />);
      expect(screen.getByText("No assignments yet")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Your section leader will add assignments before rehearsal"
        )
      ).toBeInTheDocument();
    });

    it("does not show action by default", () => {
      renderWithRouter(<NoAssignmentsEmptyState />);
      expect(
        screen.queryByRole("button", { name: "Create Assignment" })
      ).not.toBeInTheDocument();
    });

    it("shows action when showAction is true", async () => {
      const handleAction = vi.fn();
      renderWithRouter(
        <NoAssignmentsEmptyState showAction onAction={handleAction} />
      );

      const button = screen.getByRole("button", { name: "Create Assignment" });
      await userEvent.click(button);
      expect(handleAction).toHaveBeenCalledTimes(1);
    });
  });

  describe("NoTicketsEmptyState", () => {
    it("renders correct title and description", () => {
      renderWithRouter(<NoTicketsEmptyState />);
      expect(screen.getByText("All clear!")).toBeInTheDocument();
      expect(
        screen.getByText("No tickets to work on right now")
      ).toBeInTheDocument();
    });
  });

  describe("NoPracticeLogsEmptyState", () => {
    it("renders correct title and description", () => {
      renderWithRouter(<NoPracticeLogsEmptyState />);
      expect(screen.getByText("Start your streak!")).toBeInTheDocument();
      expect(
        screen.getByText("Log your first practice session")
      ).toBeInTheDocument();
    });

    it("shows Log Practice action when onAction provided", async () => {
      const handleAction = vi.fn();
      renderWithRouter(<NoPracticeLogsEmptyState onAction={handleAction} />);

      const button = screen.getByRole("button", { name: "Log Practice" });
      await userEvent.click(button);
      expect(handleAction).toHaveBeenCalledTimes(1);
    });
  });

  describe("NoTeamMembersEmptyState", () => {
    it("renders correct title", () => {
      renderWithRouter(<NoTeamMembersEmptyState />);
      expect(screen.getByText("Invite your team")).toBeInTheDocument();
    });

    it("shows Create Invite action when showAction is true", async () => {
      const handleAction = vi.fn();
      renderWithRouter(
        <NoTeamMembersEmptyState showAction onAction={handleAction} />
      );

      const button = screen.getByRole("button", { name: "Create Invite" });
      await userEvent.click(button);
      expect(handleAction).toHaveBeenCalledTimes(1);
    });
  });

  describe("NoFilterResultsEmptyState", () => {
    it("renders correct title and description", () => {
      renderWithRouter(<NoFilterResultsEmptyState />);
      expect(screen.getByText("No matches")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Try adjusting your filters to find what you're looking for"
        )
      ).toBeInTheDocument();
    });

    it("shows Clear filters action when onAction provided", async () => {
      const handleAction = vi.fn();
      renderWithRouter(<NoFilterResultsEmptyState onAction={handleAction} />);

      const button = screen.getByRole("button", { name: "Clear filters" });
      await userEvent.click(button);
      expect(handleAction).toHaveBeenCalledTimes(1);
    });
  });

  describe("NoSearchResultsEmptyState", () => {
    it("renders correct title", () => {
      renderWithRouter(<NoSearchResultsEmptyState />);
      expect(screen.getByText("No results found")).toBeInTheDocument();
    });
  });

  describe("NoCycleEmptyState", () => {
    it("renders correct title and description", () => {
      renderWithRouter(<NoCycleEmptyState />);
      expect(screen.getByText("No Rehearsal Scheduled")).toBeInTheDocument();
    });
  });

  describe("NoInvitesEmptyState", () => {
    it("renders correct title", () => {
      renderWithRouter(<NoInvitesEmptyState />);
      expect(screen.getByText("No active invites")).toBeInTheDocument();
    });
  });

  describe("NoTicketsToVerifyEmptyState", () => {
    it("renders correct title", () => {
      renderWithRouter(<NoTicketsToVerifyEmptyState />);
      expect(
        screen.getByText("No tickets pending verification")
      ).toBeInTheDocument();
    });

    it("uses compact styling", () => {
      const { container } = renderWithRouter(<NoTicketsToVerifyEmptyState />);
      // Compact version uses py-6
      expect(container.querySelector(".py-6")).toBeInTheDocument();
    });
  });
});

