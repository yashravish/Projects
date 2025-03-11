import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should allow user to register', async ({ page }) => {
    await page.goto('/register');
    
    const email = `test${Date.now()}@example.com`;
    await page.fill('input[name="email"]', email);
    await page.fill('input[name="username"]', 'testuser');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // Should redirect to home page after successful registration
    await expect(page).toHaveURL('/');
  });

  test('should allow user to login', async ({ page }) => {
    await page.goto('/login');
    
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // Should redirect to home page after successful login
    await expect(page).toHaveURL('/');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    await page.fill('input[name="email"]', 'wrong@example.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    // Should show error message
    await expect(page.locator('.text-red-300')).toBeVisible();
  });

  test('should prevent duplicate usernames during registration', async ({ page }) => {
    await page.goto('/register');
    
    await page.fill('input[name="email"]', 'new@example.com');
    await page.fill('input[name="username"]', 'testuser'); // Using existing username
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // Should show error message
    await expect(page.locator('.text-red-500')).toBeVisible();
  });
});

test.describe('Posts', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto('/login');
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');
  });

  test('should create a new post', async ({ page }) => {
    await page.goto('/create-post');
    
    const title = `Test Post ${Date.now()}`;
    await page.fill('input[name="title"]', title);
    await page.fill('textarea[name="content"]', 'This is a test post content');
    await page.click('button[type="submit"]');

    // Should redirect to the new post page
    await expect(page.locator('h1')).toContainText(title);
  });

  test('should allow voting on posts', async ({ page }) => {
    await page.goto('/');

    // Initial vote count
    const initialCount = await page.locator('.neon-text').first().innerText();
    
    // Click upvote button on first post
    await page.click('button[aria-label="Upvote"]');

    // Wait for vote count to update
    await expect(async () => {
      const newCount = await page.locator('.neon-text').first().innerText();
      expect(Number(newCount)).toBe(Number(initialCount) + 1);
    }).toPass();

    // Click downvote button
    await page.click('button[aria-label="Downvote"]');

    // Wait for vote count to update
    await expect(async () => {
      const newCount = await page.locator('.neon-text').first().innerText();
      expect(Number(newCount)).toBe(Number(initialCount) - 1);
    }).toPass();
  });

  test('should allow commenting on posts', async ({ page }) => {
    // Go to first post
    await page.goto('/');
    await page.click('h2');

    const comment = `Test comment ${Date.now()}`;
    await page.fill('textarea', comment);
    await page.click('button[type="submit"]');

    // Wait for comment to appear
    await expect(page.locator('.text-gray-700')).toContainText(comment);
  });

  test('should load more posts on scroll', async ({ page }) => {
    await page.goto('/');

    // Get initial post count
    const initialPosts = await page.locator('.glass-card').count();

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));

    // Wait for more posts to load
    await expect(async () => {
      const newPostCount = await page.locator('.glass-card').count();
      expect(newPostCount).toBeGreaterThan(initialPosts);
    }).toPass();
  });
});