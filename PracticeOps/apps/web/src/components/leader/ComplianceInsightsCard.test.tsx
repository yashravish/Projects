import { describe, expect, it } from "vitest";
import { buildComplianceChartData } from "./ComplianceInsightsCard";

describe("buildComplianceChartData", () => {
  it("sorts sections by average days descending", () => {
    const data = buildComplianceChartData([
      {
        section: "Tenor",
        member_count: 2,
        total_practice_days_7d: 4,
        avg_practice_days_7d: 2,
      },
      {
        section: "Soprano",
        member_count: 3,
        total_practice_days_7d: 12,
        avg_practice_days_7d: 4,
      },
    ]);

    expect(data[0].name).toBe("Soprano");
    expect(data[1].name).toBe("Tenor");
  });

  it("maps section metrics to chart data", () => {
    const data = buildComplianceChartData([
      {
        section: "Bass",
        member_count: 4,
        total_practice_days_7d: 8,
        avg_practice_days_7d: 2.5,
      },
    ]);

    expect(data[0]).toEqual({
      name: "Bass",
      avgDays: 2.5,
      members: 4,
      totalDays: 8,
    });
  });
});
