import { describe, it, expect } from 'vitest';
import { hashPassword, verifyPassword, generateToken } from '../services/auth.service';

describe('Auth Service', () => {
  it('should hash password', async () => {
    const password = 'testpassword';
    const hash = await hashPassword(password);

    expect(hash).toBeDefined();
    expect(hash).not.toBe(password);
    expect(hash.length).toBeGreaterThan(0);
  });

  it('should verify correct password', async () => {
    const password = 'testpassword';
    const hash = await hashPassword(password);
    const isValid = await verifyPassword(password, hash);

    expect(isValid).toBe(true);
  });

  it('should reject incorrect password', async () => {
    const password = 'testpassword';
    const hash = await hashPassword(password);
    const isValid = await verifyPassword('wrongpassword', hash);

    expect(isValid).toBe(false);
  });

  it('should generate JWT token', () => {
    const user = {
      userId: 'test-user-id',
      email: 'test@example.com',
    };

    const token = generateToken(user);

    expect(token).toBeDefined();
    expect(typeof token).toBe('string');
    expect(token.split('.')).toHaveLength(3);
  });
});
