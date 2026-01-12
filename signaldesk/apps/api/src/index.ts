import Fastify from 'fastify';
import cors from '@fastify/cors';
import cookie from '@fastify/cookie';
import multipart from '@fastify/multipart';
import helmet from '@fastify/helmet';
import { config } from './config';
import { authRoutes } from './routes/auth.routes';
import { collectionsRoutes } from './routes/collections.routes';
import { documentsRoutes } from './routes/documents.routes';
import { queryRoutes } from './routes/query.routes';
import { analyticsRoutes } from './routes/analytics.routes';
import './jobs/worker';

const fastify = Fastify({
  logger: config.nodeEnv === 'development',
});

async function start() {
  try {
    await fastify.register(helmet, {
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", 'data:', 'https:'],
        },
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true,
      },
    });

    await fastify.register(cors, {
      origin: (origin, cb) => {
        // Allow requests with no origin (like mobile apps or curl requests)
        if (!origin) {
          cb(null, true);
          return;
        }

        // Check if origin matches configured CORS origins
        const allowedOrigins = config.corsOrigins;

        // Check if origin is in the allowed list
        if (allowedOrigins.includes(origin)) {
          cb(null, true);
          return;
        }

        // Allow all Vercel preview and production domains
        if (origin.endsWith('.vercel.app')) {
          cb(null, true);
          return;
        }

        // Reject other origins
        cb(new Error('Not allowed by CORS'), false);
      },
      credentials: true,
    });

    await fastify.register(cookie);

    await fastify.register(multipart, {
      limits: {
        fileSize: 10 * 1024 * 1024,
      },
    });

    fastify.get('/health', async () => {
      return { status: 'ok', timestamp: new Date().toISOString() };
    });

    await fastify.register(authRoutes, { prefix: '/v1/auth' });
    await fastify.register(collectionsRoutes, { prefix: '/v1/collections' });
    await fastify.register(documentsRoutes, { prefix: '/v1' });
    await fastify.register(queryRoutes, { prefix: '/v1' });
    await fastify.register(analyticsRoutes, { prefix: '/v1' });

    fastify.setErrorHandler((error, _request, reply) => {
      console.error(error);

      const statusCode = error.statusCode || 500;
      const code = error.name || 'INTERNAL_SERVER_ERROR';
      const message = error.message || 'An unexpected error occurred';

      reply.status(statusCode).send({
        error: {
          code,
          message,
        },
      });
    });

    await fastify.listen({
      port: config.port,
      host: config.host,
    });

    console.log(`🚀 API server running at http://${config.host}:${config.port}`);
  } catch (error) {
    console.error('Error starting server:', error);
    process.exit(1);
  }
}

start();
