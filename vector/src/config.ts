import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs/promises';
import { z } from 'zod';

export const ConfigSchema = z.object({
  embeddingModel: z.string().default('all-MiniLM-L6-v2'),
  modelPath: z.string().default(path.join(os.homedir(), '.vdiagram', 'models', 'all-MiniLM-L6-v2')),
  indexPath: z.string().default(path.join('.vdiagram', 'index')),
  dbPath: z.string().default(path.join('.vdiagram', 'db')),
  defaultK: z.number().int().positive().default(24),
  mmrLambda: z.number().min(0).max(1).default(0.5),
  minScore: z.number().min(0).max(1).default(0.2),
  seed: z.number().int().default(1337),
  ignoreGlobs: z.array(z.string()).default([
    'node_modules/**',
    '.git/**',
    '**/*.png',
    '**/*.jpg',
    '**/*.jpeg',
    '**/*.gif',
    '**/*.pdf',
    '**/*.zip',
  ]),
  tags: z.array(z.string()).default([]),
});

export type VDiagramConfig = z.infer<typeof ConfigSchema>;

export async function resolveConfigPath(provided?: string): Promise<string> {
  if (provided) return path.resolve(provided);
  return path.resolve('.vdiagram.json');
}

export async function writeDefaultConfig(configPath: string): Promise<VDiagramConfig> {
  const defaults = ConfigSchema.parse({});
  const content = JSON.stringify(defaults, null, 2);
  await fs.writeFile(configPath, content, { encoding: 'utf8' });
  return defaults;
}

export async function loadConfig(configPath: string): Promise<VDiagramConfig> {
  try {
    const raw = await fs.readFile(configPath, 'utf8');
    const parsed = JSON.parse(raw);
    // Merge with defaults using zod
    const cfg = ConfigSchema.parse(parsed);
    return cfg;
  } catch (err: any) {
    if (err.code === 'ENOENT') {
      // If not found, create defaults
      return writeDefaultConfig(configPath);
    }
    throw err;
  }
}

export async function ensureProjectDirs(cfg: VDiagramConfig): Promise<void> {
  const dirs = [cfg.modelPath, cfg.indexPath, cfg.dbPath];
  for (const d of dirs) {
    const absolute = expandHome(d);
    await fs.mkdir(absolute, { recursive: true });
  }
}

export function expandHome(p: string): string {
  if (p.startsWith('~')) {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}


