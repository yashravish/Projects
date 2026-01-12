import { FastifyRequest, FastifyReply } from 'fastify';
import jwt from 'jsonwebtoken';
import { config } from '../config';

export interface AuthUser {
  userId: string;
  email: string;
}

declare module 'fastify' {
  interface FastifyRequest {
    user?: AuthUser;
  }
}

export async function authMiddleware(request: FastifyRequest, reply: FastifyReply) {
  try {
    const token = request.cookies.token;

    if (!token) {
      return reply.status(401).send({
        error: {
          code: 'UNAUTHORIZED',
          message: 'No authentication token provided',
        },
      });
    }

    const decoded = jwt.verify(token, config.jwtSecret) as AuthUser;
    request.user = decoded;
  } catch (error) {
    return reply.status(401).send({
      error: {
        code: 'UNAUTHORIZED',
        message: 'Invalid or expired token',
      },
    });
  }
}
