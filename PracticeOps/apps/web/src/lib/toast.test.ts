/**
 * Toast Utility Tests
 *
 * Tests for toast notification utilities.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { toastSuccess, toastError, toastWarning, toastInfo } from "./toast";

// Mock the toast function from use-toast
vi.mock("@/hooks/use-toast", () => ({
  toast: vi.fn((options) => ({
    id: "test-id",
    dismiss: vi.fn(),
    update: vi.fn(),
    ...options,
  })),
}));

// Import the mocked toast
import { toast } from "@/hooks/use-toast";

describe("Toast Utilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("toastSuccess", () => {
    it("calls toast with success styling", () => {
      toastSuccess("Operation completed");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Success",
          description: "Operation completed",
        })
      );
    });

    it("includes green border styling", () => {
      toastSuccess("Test message");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          className: expect.stringContaining("border-green"),
        })
      );
    });

    it("accepts custom title", () => {
      toastSuccess("Test message", { title: "Custom Title" });

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Custom Title",
        })
      );
    });
  });

  describe("toastError", () => {
    it("calls toast with destructive variant", () => {
      toastError("Something went wrong");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          title: "Error",
          description: "Something went wrong",
        })
      );
    });

    it("accepts custom title", () => {
      toastError("Test message", { title: "Failed" });

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed",
        })
      );
    });
  });

  describe("toastWarning", () => {
    it("calls toast with warning styling", () => {
      toastWarning("Heads up!");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Warning",
          description: "Heads up!",
        })
      );
    });

    it("includes amber border styling", () => {
      toastWarning("Test message");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          className: expect.stringContaining("border-amber"),
        })
      );
    });
  });

  describe("toastInfo", () => {
    it("calls toast with info styling", () => {
      toastInfo("FYI");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Info",
          description: "FYI",
        })
      );
    });

    it("includes blue border styling", () => {
      toastInfo("Test message");

      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          className: expect.stringContaining("border-blue"),
        })
      );
    });
  });
});

