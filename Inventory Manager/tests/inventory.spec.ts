import { test, expect } from '@playwright/test';

test.describe('Inventory Management System', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display products on the home page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Available Products');
    await expect(page.getByRole('article')).toHaveCount(1);
  });

  test('should handle product stock updates correctly', async ({ page }) => {
    // Add product to cart
    const firstProduct = page.getByRole('article').first();
    const initialStock = await firstProduct.locator('.text-gray-500').textContent();
    await firstProduct.getByRole('button', { name: 'Add to Cart' }).click();
    
    // Complete checkout
    await page.getByRole('link', { name: 'Cart' }).click();
    await page.getByRole('button', { name: 'Checkout' }).click();
    
    // Verify stock was decremented
    await page.getByRole('link', { name: 'Products' }).click();
    const updatedStock = await firstProduct.locator('.text-gray-500').textContent();
    expect(Number(updatedStock)).toBeLessThan(Number(initialStock));
  });

  test('admin can manage products', async ({ page }) => {
    // Login as admin
    await page.getByRole('link', { name: 'Admin' }).click();
    
    // Add new product
    await page.getByLabel('Name').fill('Test Product');
    await page.getByLabel('Price').fill('99.99');
    await page.getByLabel('Stock').fill('10');
    await page.getByRole('button', { name: 'Add Product' }).click();
    
    // Verify product was added
    await expect(page.getByText('Test Product')).toBeVisible();
    await expect(page.getByText('$99.99')).toBeVisible();
  });

  test('should prevent stock going negative', async ({ page }) => {
    // Add product with more quantity than available
    const product = page.getByRole('article').first();
    const stock = await product.locator('.text-gray-500').textContent();
    const quantity = Number(stock) + 1;
    
    // Try to checkout
    await page.getByRole('link', { name: 'Cart' }).click();
    await page.getByRole('button', { name: 'Checkout' }).click();
    
    // Verify error message
    await expect(page.getByText('Insufficient stock')).toBeVisible();
  });
});