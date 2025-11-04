/**
 * Tests for swara mapping
 */

import { describe, it, expect } from "vitest";
import {
  centsToSwara,
  getSwaraCents,
  swaraInterval,
  isNearSwara,
  SWARA_CENTS,
  ALL_SWARAS,
} from "../src/theory/swara-map";
import { getRaga } from "../src/theory/ragas";

describe("Swara Mapping", () => {
  describe("centsToSwara", () => {
    it("should map 0 cents to S (Sa)", () => {
      const result = centsToSwara(0);

      expect(result).not.toBeNull();
      expect(result!.swara).toBe("S");
      expect(result!.confidence).toBeGreaterThan(0.9);
    });

    it("should map 204 cents to R2 (Chathusruthi Rishabham)", () => {
      const result = centsToSwara(204);

      expect(result).not.toBeNull();
      expect(result!.swara).toBe("R2");
    });

    it("should map 702 cents to P (Panchamam)", () => {
      const result = centsToSwara(702);

      expect(result).not.toBeNull();
      expect(result!.swara).toBe("P");
      expect(result!.confidence).toBeGreaterThan(0.9);
    });

    it("should handle octave wrapping (1200 cents = S)", () => {
      const result = centsToSwara(1200);

      expect(result).not.toBeNull();
      expect(result!.swara).toBe("S");
    });

    it("should handle negative cents", () => {
      const result = centsToSwara(-204); // One octave below R2

      expect(result).not.toBeNull();
      expect(result!.swara).toBe("N2");
    });

    it("should return null for cents far from any swara", () => {
      const result = centsToSwara(50, ALL_SWARAS, 30); // 50 cents, tolerance 30

      // 50 cents is between S (0) and R1 (90), but beyond tolerance
      expect(result).toBeNull();
    });

    it("should respect raga constraints (Mohanam)", () => {
      const raga = getRaga("Mohanam");
      expect(raga).not.toBeNull();

      const allowedSwaras = raga!.allowedSwaras;

      // Try to map to R1 (90 cents), which is not in Mohanam
      const result1 = centsToSwara(90, allowedSwaras);
      expect(result1).toBeNull(); // R1 not allowed

      // Try R2 (204 cents), which is in Mohanam
      const result2 = centsToSwara(204, allowedSwaras);
      expect(result2).not.toBeNull();
      expect(result2!.swara).toBe("R2");
    });

    it("should have lower confidence for cents farther from exact swara", () => {
      const exact = centsToSwara(204); // Exactly R2
      const close = centsToSwara(210); // Close to R2
      const farther = centsToSwara(220); // Farther from R2

      expect(exact!.confidence).toBeGreaterThan(close!.confidence);
      expect(close!.confidence).toBeGreaterThan(farther!.confidence);
    });
  });

  describe("getSwaraCents", () => {
    it("should return correct cents for each swara", () => {
      expect(getSwaraCents("S")).toBe(0);
      expect(getSwaraCents("R2")).toBe(204);
      expect(getSwaraCents("G3")).toBe(386);
      expect(getSwaraCents("M1")).toBe(498);
      expect(getSwaraCents("P")).toBe(702);
      expect(getSwaraCents("D2")).toBe(906);
      expect(getSwaraCents("N3")).toBe(1088);
    });
  });

  describe("swaraInterval", () => {
    it("should calculate interval from S to P", () => {
      const interval = swaraInterval("S", "P");
      expect(interval).toBe(702);
    });

    it("should calculate interval from R2 to G3", () => {
      const interval = swaraInterval("R2", "G3");
      expect(interval).toBe(386 - 204);
    });

    it("should handle descending intervals (negative)", () => {
      const interval = swaraInterval("P", "S");
      expect(interval).toBe(-702);
    });
  });

  describe("isNearSwara", () => {
    it("should detect cents near S", () => {
      expect(isNearSwara(0, "S", 25)).toBe(true);
      expect(isNearSwara(10, "S", 25)).toBe(true);
      expect(isNearSwara(50, "S", 25)).toBe(false);
    });

    it("should detect cents near P", () => {
      expect(isNearSwara(702, "P", 25)).toBe(true);
      expect(isNearSwara(710, "P", 25)).toBe(true);
      expect(isNearSwara(750, "P", 25)).toBe(false);
    });

    it("should handle octave wrapping", () => {
      expect(isNearSwara(1200, "S", 25)).toBe(true);
      expect(isNearSwara(1210, "S", 25)).toBe(true);
    });
  });

  describe("SWARA_CENTS constants", () => {
    it("should have all swaras defined", () => {
      expect(SWARA_CENTS).toHaveProperty("S");
      expect(SWARA_CENTS).toHaveProperty("R1");
      expect(SWARA_CENTS).toHaveProperty("R2");
      expect(SWARA_CENTS).toHaveProperty("G2");
      expect(SWARA_CENTS).toHaveProperty("G3");
      expect(SWARA_CENTS).toHaveProperty("M1");
      expect(SWARA_CENTS).toHaveProperty("M2");
      expect(SWARA_CENTS).toHaveProperty("P");
      expect(SWARA_CENTS).toHaveProperty("D1");
      expect(SWARA_CENTS).toHaveProperty("D2");
      expect(SWARA_CENTS).toHaveProperty("N2");
      expect(SWARA_CENTS).toHaveProperty("N3");
    });

    it("should have cents in ascending order", () => {
      const cents = Object.values(SWARA_CENTS);
      for (let i = 1; i < cents.length; i++) {
        expect(cents[i]).toBeGreaterThan(cents[i - 1]);
      }
    });

    it("should be within octave range (0-1200)", () => {
      for (const cents of Object.values(SWARA_CENTS)) {
        expect(cents).toBeGreaterThanOrEqual(0);
        expect(cents).toBeLessThan(1200);
      }
    });
  });
});
