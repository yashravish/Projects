import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { CONFIG } from '@signaldesk/shared';
import { config } from '../config';
import { AuthUser } from '../middleware/auth';

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, CONFIG.BCRYPT_ROUNDS);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export function generateToken(user: AuthUser): string {
  return jwt.sign(user, config.jwtSecret, {
    expiresIn: CONFIG.JWT_EXPIRY,
  });
}
