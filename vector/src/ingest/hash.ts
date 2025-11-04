import crypto from 'node:crypto';
import fs from 'node:fs/promises';

export type FileHashInfo = {
  algorithm: 'sha256';
  hashHex: string;
  size: number;
  mtimeMs: number;
};

export async function hashFile(path: string): Promise<FileHashInfo> {
  const stat = await fs.stat(path);
  const data = await fs.readFile(path);
  const hash = crypto.createHash('sha256').update(data).digest('hex');
  return { algorithm: 'sha256', hashHex: hash, size: stat.size, mtimeMs: stat.mtimeMs };
}

export function hashString(text: string): string {
  return crypto.createHash('sha256').update(text).digest('hex');
}


