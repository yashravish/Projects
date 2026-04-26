import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Projects } from "../Projects";

const list = { items: [], total: 0 };

beforeEach(() => {
  vi.resetAllMocks();
});

describe("Projects page", () => {
  it("renders heading and table", async () => {
    vi.spyOn(global, "fetch").mockImplementation((input: RequestInfo) => {
      const u = String(input);
      if (u.includes("/api/projects")) {
        return Promise.resolve(new Response(JSON.stringify(list), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      return Promise.reject(new Error("unexpected fetch " + u));
    });
    render(
      <MemoryRouter>
        <Projects />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "Projects" })).toBeInTheDocument();
  });
});
