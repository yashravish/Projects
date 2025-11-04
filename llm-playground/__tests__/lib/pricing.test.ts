import { estimateCostUSD, roughTokenEstimate } from "@/lib/pricing";

describe("pricing", () => {
  describe("roughTokenEstimate", () => {
    it("estimates tokens from text", () => {
      expect(roughTokenEstimate("Hello world")).toBe(3);
    });

    it("returns at least 1 for empty text", () => {
      expect(roughTokenEstimate("")).toBe(1);
    });

    it("trims whitespace before estimating", () => {
      expect(roughTokenEstimate("  test  ")).toBe(1);
    });
  });

  describe("estimateCostUSD", () => {
    it("calculates cost for gpt-4o-mini", () => {
      const cost = estimateCostUSD("gpt-4o-mini", 1000, 500);
      expect(cost).toBeCloseTo(0.00045, 6);
    });

    it("uses fallback pricing for unknown models", () => {
      const cost = estimateCostUSD("unknown-model", 1000, 500);
      expect(cost).toBeGreaterThan(0);
    });
  });
});
