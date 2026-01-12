import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../.env') });

if (!process.env.JWT_SECRET) {
  throw new Error('JWT_SECRET environment variable is required for security. Please set it in your .env file.');
}

export const config = {
  port: parseInt(process.env.API_PORT || '3001', 10),
  host: process.env.API_HOST || '0.0.0.0',
  nodeEnv: process.env.NODE_ENV || 'development',
  jwtSecret: process.env.JWT_SECRET,
  openaiApiKey: process.env.OPENAI_API_KEY || '',
  databaseUrl: process.env.DATABASE_URL || '',
  redisUrl: process.env.REDIS_URL || '',
  upstashRedisRestUrl: process.env.UPSTASH_REDIS_REST_URL || '',
  upstashRedisRestToken: process.env.UPSTASH_REDIS_REST_TOKEN || '',
  corsOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000'],
};

if (!config.openaiApiKey) {
  console.warn('WARNING: OPENAI_API_KEY not set. Running in stub mode.');
}
