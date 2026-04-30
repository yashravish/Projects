import { expect, test } from '@playwright/test';

const ANALYST = { email: 'analyst@example.com', password: 'password123' };
const MANAGER = { email: 'manager@example.com', password: 'password123' };

async function login(page, creds: { email: string; password: string }) {
  await page.goto('/login');
  await page.getByLabel('Email').fill(creds.email);
  await page.getByLabel('Password').fill(creds.password);
  await page.getByRole('button', { name: /sign in/i }).click();
  await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible({
    timeout: 15_000,
  });
}

test.describe('Business Workflow end-to-end', () => {
  test('analyst logs in and creates a task', async ({ page }) => {
    await login(page, ANALYST);
    await expect(page).toHaveURL(/\/dashboard/);

    const uniqueTitle = `E2E task ${Date.now()}`;

    await page.getByRole('link', { name: 'Create Task', exact: true }).click();
    await page.getByLabel('Title').fill(uniqueTitle);
    await page.getByLabel('Description').fill('Created by Playwright e2e suite.');
    await page.getByLabel('Priority').selectOption('high');
    // Pick the first non-empty option in the assignee dropdown.
    const assigneeSelect = page.getByLabel('Assigned to');
    const firstUserOption = await assigneeSelect.locator('option').nth(1).getAttribute('value');
    if (firstUserOption) {
      await assigneeSelect.selectOption(firstUserOption);
    }
    await page.getByRole('button', { name: /create task/i }).click();
    await expect(page.getByRole('heading', { name: uniqueTitle })).toBeVisible({
      timeout: 15_000,
    });

    await page.getByTestId('submit-task').click();
    await expect(page.getByText('Submitted', { exact: true }).first()).toBeVisible({
      timeout: 15_000,
    });
  });

  test('manager approves a submitted task and dashboard updates', async ({ page }) => {
    await login(page, MANAGER);

    await page.getByRole('link', { name: 'Tasks', exact: true }).click();
    await page.locator('#filter-status').selectOption('submitted');

    const firstSubmittedRow = page.getByTestId('task-row').first();
    await expect(firstSubmittedRow).toBeVisible({ timeout: 15_000 });
    await firstSubmittedRow.getByRole('link').first().click();

    await page.getByTestId('approve-task').click();
    await expect(page.getByText('Approved', { exact: true }).first()).toBeVisible({
      timeout: 15_000,
    });
  });

  test('reports page loads and CSV export button is present', async ({ page }) => {
    await login(page, MANAGER);

    await page.getByRole('link', { name: 'Reports', exact: true }).click();
    await expect(page.getByRole('heading', { name: /reports/i })).toBeVisible();
    await expect(page.getByTestId('export-csv')).toBeVisible();

    const downloadPromise = page.waitForEvent('download');
    await page.getByTestId('export-csv').click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('tasks.csv');
  });

  test('audit log page renders entries', async ({ page }) => {
    await login(page, MANAGER);
    await page.getByRole('link', { name: 'Audit Logs', exact: true }).click();
    await expect(page.getByRole('heading', { name: /audit logs/i })).toBeVisible();
  });
});
