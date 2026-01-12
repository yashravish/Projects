import { test, expect } from "@playwright/test";

test("homepage has title", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /know where your ensemble stands/i })
  ).toBeVisible();
});

test("homepage has subtitle", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByText(/members log practice/i)
  ).toBeVisible();
});

