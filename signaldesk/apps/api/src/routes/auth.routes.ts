import { FastifyInstance } from 'fastify';
import rateLimit from '@fastify/rate-limit';
import { signupSchema, loginSchema } from '@signaldesk/shared';
import { prisma } from '../db';
import { hashPassword, verifyPassword, generateToken } from '../services/auth.service';
import { authMiddleware } from '../middleware/auth';

export async function authRoutes(fastify: FastifyInstance) {
  await fastify.register(rateLimit, {
    max: 5,
    timeWindow: '15 minutes',
    errorResponseBuilder: () => ({
      error: {
        code: 'RATE_LIMIT_EXCEEDED',
        message: 'Too many requests. Please try again later.',
      },
    }),
  });

  fastify.post('/signup', async (request, reply) => {
    try {
      const body = signupSchema.parse(request.body);

      const existingUser = await prisma.user.findUnique({
        where: { email: body.email },
      });

      if (existingUser) {
        return reply.status(409).send({
          error: {
            code: 'CONFLICT',
            message: 'User with this email already exists',
          },
        });
      }

      const passwordHash = await hashPassword(body.password);

      const user = await prisma.user.create({
        data: {
          email: body.email,
          passwordHash,
        },
      });

      const token = generateToken({ userId: user.id, email: user.email });

      reply.setCookie('token', token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
        maxAge: 7 * 24 * 60 * 60,
        path: '/',
      });

      return reply.status(201).send({
        data: {
          user: {
            id: user.id,
            email: user.email,
            createdAt: user.createdAt,
          },
        },
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid input',
            details: error,
          },
        });
      }
      throw error;
    }
  });

  fastify.post('/login', async (request, reply) => {
    try {
      const body = loginSchema.parse(request.body);

      const user = await prisma.user.findUnique({
        where: { email: body.email },
      });

      if (!user || !(await verifyPassword(body.password, user.passwordHash))) {
        return reply.status(401).send({
          error: {
            code: 'UNAUTHORIZED',
            message: 'Invalid email or password',
          },
        });
      }

      const token = generateToken({ userId: user.id, email: user.email });

      reply.setCookie('token', token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
        maxAge: 7 * 24 * 60 * 60,
        path: '/',
      });

      return reply.send({
        data: {
          user: {
            id: user.id,
            email: user.email,
            createdAt: user.createdAt,
          },
        },
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'ZodError') {
        return reply.status(400).send({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Invalid input',
            details: error,
          },
        });
      }
      throw error;
    }
  });

  fastify.post('/logout', { preHandler: authMiddleware }, async (_request, reply) => {
    reply.clearCookie('token', { path: '/' });
    return reply.send({ data: { message: 'Logged out successfully' } });
  });

  fastify.get('/me', { preHandler: authMiddleware }, async (request, reply) => {
    const user = await prisma.user.findUnique({
      where: { id: request.user!.userId },
    });

    if (!user) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'User not found',
        },
      });
    }

    return reply.send({
      data: {
        user: {
          id: user.id,
          email: user.email,
          createdAt: user.createdAt,
        },
      },
    });
  });
}
