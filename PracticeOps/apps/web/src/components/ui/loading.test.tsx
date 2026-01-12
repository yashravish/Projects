/**
 * Loading Components Tests
 *
 * Tests for loading state components:
 * - PageLoader
 * - ListSkeleton
 * - ButtonSpinner
 * - InlineLoader
 * - DashboardSkeleton
 * - TableSkeleton
 * - CardSkeleton
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  PageLoader,
  ListSkeleton,
  ButtonSpinner,
  InlineLoader,
  DashboardSkeleton,
  TableSkeleton,
  CardSkeleton,
} from "./loading";

describe("Loading Components", () => {
  describe("PageLoader", () => {
    it("renders with default message", () => {
      render(<PageLoader />);
      expect(screen.getByText("Loading...")).toBeInTheDocument();
    });

    it("renders with custom message", () => {
      render(<PageLoader message="Fetching data..." />);
      expect(screen.getByText("Fetching data...")).toBeInTheDocument();
    });

    it("applies custom className", () => {
      const { container } = render(<PageLoader className="custom-class" />);
      expect(container.firstChild).toHaveClass("custom-class");
    });
  });

  describe("ListSkeleton", () => {
    it("renders default 3 card skeletons", () => {
      const { container } = render(<ListSkeleton />);
      const skeletons = container.querySelectorAll(".rounded-lg.border");
      expect(skeletons.length).toBe(3);
    });

    it("renders specified number of skeletons", () => {
      const { container } = render(<ListSkeleton count={5} />);
      const skeletons = container.querySelectorAll(".rounded-lg.border");
      expect(skeletons.length).toBe(5);
    });

    it("renders row variant", () => {
      const { container } = render(<ListSkeleton variant="row" count={3} />);
      const rows = container.querySelectorAll(".flex.items-center.gap-4");
      expect(rows.length).toBe(3);
    });
  });

  describe("ButtonSpinner", () => {
    it("renders with animate-spin class", () => {
      const { container } = render(<ButtonSpinner />);
      const spinner = container.querySelector("svg");
      expect(spinner).toHaveClass("animate-spin");
    });

    it("applies size classes", () => {
      const { container: smContainer } = render(<ButtonSpinner size="sm" />);
      const smSpinner = smContainer.querySelector("svg");
      expect(smSpinner).toHaveClass("h-3", "w-3");

      const { container: lgContainer } = render(<ButtonSpinner size="lg" />);
      const lgSpinner = lgContainer.querySelector("svg");
      expect(lgSpinner).toHaveClass("h-5", "w-5");
    });
  });

  describe("InlineLoader", () => {
    it("renders with animate-spin class", () => {
      const { container } = render(<InlineLoader />);
      const spinner = container.querySelector("svg");
      expect(spinner).toHaveClass("animate-spin");
    });

    it("has muted text color", () => {
      const { container } = render(<InlineLoader />);
      const spinner = container.querySelector("svg");
      expect(spinner).toHaveClass("text-muted-foreground");
    });
  });

  describe("DashboardSkeleton", () => {
    it("renders stats row skeletons", () => {
      const { container } = render(<DashboardSkeleton />);
      // Check for grid structure
      const grid = container.querySelector(".grid.gap-4.md\\:grid-cols-3");
      expect(grid).toBeInTheDocument();
    });

    it("renders two-column layout skeleton", () => {
      const { container } = render(<DashboardSkeleton />);
      const twoColGrid = container.querySelector(".grid.gap-6.lg\\:grid-cols-3");
      expect(twoColGrid).toBeInTheDocument();
    });
  });

  describe("TableSkeleton", () => {
    it("renders default 5 rows", () => {
      const { container } = render(<TableSkeleton />);
      // Count row containers (excluding header)
      const rows = container.querySelectorAll(".flex.items-center.gap-4.py-3");
      expect(rows.length).toBe(5);
    });

    it("renders specified number of rows", () => {
      const { container } = render(<TableSkeleton rows={3} />);
      const rows = container.querySelectorAll(".flex.items-center.gap-4.py-3");
      expect(rows.length).toBe(3);
    });

    it("renders header row", () => {
      const { container } = render(<TableSkeleton />);
      const header = container.querySelector(".flex.items-center.gap-4.border-b");
      expect(header).toBeInTheDocument();
    });
  });

  describe("CardSkeleton", () => {
    it("renders card structure", () => {
      const { container } = render(<CardSkeleton />);
      expect(container.querySelector(".rounded-lg.border")).toBeInTheDocument();
    });

    it("applies custom className", () => {
      const { container } = render(<CardSkeleton className="custom-class" />);
      expect(container.firstChild).toHaveClass("custom-class");
    });
  });
});

