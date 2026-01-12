import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import * as authService from '../services/auth.service';
import { prisma } from '../db';

// Mock dependencies
vi.mock('../db', () => ({
  prisma: {
    user: {
      findUnique: vi.fn(),
      create: vi.fn(),
    },
  },
}));

vi.mock('../services/auth.service', () => ({
  hashPassword: vi.fn(),
  verifyPassword: vi.fn(),
  generateToken: vi.fn(),
}));

vi.mock('../middleware/auth', () => ({
  authMiddleware: async (request: any, _reply: any) => {
    request.user = { userId: 'test-user-id', email: 'test@example.com' };
  },
}));

// Mock rate-limit to avoid version mismatch
vi.mock('@fastify/rate-limit', () => ({
  default: async (fastify: any, opts: any) => {
    // No-op for tests
  },
}));

describe('Auth Routes', () => {
  let fastify: FastifyInstance;

  beforeEach(async () => {
    fastify = Fastify();
    await fastify.register(import('@fastify/cookie'));
    const { authRoutes } = await import('../routes/auth.routes');
    await fastify.register(authRoutes, { prefix: '/auth' });
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await fastify.close();
  });

  describe('POST /auth/signup', () => {
    it('should create a new user successfully', async () => {
      const mockUser = {
        id: 'new-user-id',
        email: 'newuser@example.com',
        passwordHash: 'hashed-password',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      vi.mocked(prisma.user.findUnique).mockResolvedValue(null);
      vi.mocked(authService.hashPassword).mockResolvedValue('hashed-password');
      vi.mocked(prisma.user.create).mockResolvedValue(mockUser);
      vi.mocked(authService.generateToken).mockReturnValue('test-token');

      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/signup',
        payload: {
          email: 'newuser@example.com',
          password: 'SecurePass123!',
        },
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.data.user.email).toBe('newuser@example.com');
      expect(body.data.user.id).toBe('new-user-id');
    });

    it('should return 409 if user already exists', async () => {
      vi.mocked(prisma.user.findUnique).mockResolvedValue({
        id: 'existing-user-id',
        email: 'existing@example.com',
        passwordHash: 'hash',
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/signup',
        payload: {
          email: 'existing@example.com',
          password: 'Password123!',
        },
      });

      expect(response.statusCode).toBe(409);
      const body = JSON.parse(response.body);
      expect(body.error.code).toBe('CONFLICT');
    });

    it('should return 400 for invalid email format', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/signup',
        payload: {
          email: 'invalid-email',
          password: 'Password123!',
        },
      });

      expect(response.statusCode).toBe(400);
    });

    it('should return 400 for weak password', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/signup',
        payload: {
          email: 'test@example.com',
          password: 'weak',
        },
      });

      expect(response.statusCode).toBe(400);
    });
  });

  describe('POST /auth/login', () => {
    it('should login user successfully', async () => {
      const mockUser = {
        id: 'user-id',
        email: 'user@example.com',
        passwordHash: 'hashed-password',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      vi.mocked(prisma.user.findUnique).mockResolvedValue(mockUser);
      vi.mocked(authService.verifyPassword).mockResolvedValue(true);
      vi.mocked(authService.generateToken).mockReturnValue('test-token');

      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/login',
        payload: {
          email: 'user@example.com',
          password: 'CorrectPassword123!',
        },
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.user.email).toBe('user@example.com');
    });

    it('should return 401 for non-existent user', async () => {
      vi.mocked(prisma.user.findUnique).mockResolvedValue(null);

      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/login',
        payload: {
          email: 'nonexistent@example.com',
          password: 'Password123!',
        },
      });

      expect(response.statusCode).toBe(401);
      const body = JSON.parse(response.body);
      expect(body.error.code).toBe('UNAUTHORIZED');
    });

    it('should return 401 for incorrect password', async () => {
      vi.mocked(prisma.user.findUnique).mockResolvedValue({
        id: 'user-id',
        email: 'user@example.com',
        passwordHash: 'hashed-password',
        createdAt: new Date(),
        updatedAt: new Date(),
      });
      vi.mocked(authService.verifyPassword).mockResolvedValue(false);

      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/login',
        payload: {
          email: 'user@example.com',
          password: 'WrongPassword123!',
        },
      });

      expect(response.statusCode).toBe(401);
    });
  });

  describe('POST /auth/logout', () => {
    it('should logout user successfully', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/auth/logout',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.message).toBe('Logged out successfully');
    });
  });

  describe('GET /auth/me', () => {
    it('should return current user info', async () => {
      const mockUser = {
        id: 'test-user-id',
        email: 'test@example.com',
        passwordHash: 'hash',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      vi.mocked(prisma.user.findUnique).mockResolvedValue(mockUser);

      const response = await fastify.inject({
        method: 'GET',
        url: '/auth/me',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.data.user.email).toBe('test@example.com');
    });

    it('should return 404 if user not found', async () => {
      vi.mocked(prisma.user.findUnique).mockResolvedValue(null);

      const response = await fastify.inject({
        method: 'GET',
        url: '/auth/me',
      });

      expect(response.statusCode).toBe(404);
    });
  });
});
