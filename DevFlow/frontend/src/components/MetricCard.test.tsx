import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricCard } from "./MetricCard";

describe("MetricCard", () => {
  it("renders title and value", () => {
    render(<MetricCard title="Pipelines" value={42} />);
    expect(screen.getByText("Pipelines")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });
});
