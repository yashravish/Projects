import path from 'node:path';
import fs from 'node:fs/promises';
import fg from 'fast-glob';

export type DiscoveredFile = {
  absolutePath: string;
  posixPath: string; // normalized for storage
  size: number;
  ext: string;
  langHint: string;
};

function toPosix(p: string): string {
  return p.split(path.sep).join(path.posix.sep);
}

function langFromExt(ext: string): string {
  const e = ext.toLowerCase();
  if (e === '.ts' || e === '.tsx') return 'ts';
  if (e === '.js' || e === '.jsx' || e === '.mjs' || e === '.cjs') return 'js';
  if (e === '.py') return 'py';
  if (e === '.go') return 'go';
  if (e === '.java') return 'java';
  if (e === '.md') return 'md';
  if (e === '.yaml' || e === '.yml') return 'yaml';
  if (e === '.json') return 'json';
  return e.replace(/^\./, '') || 'txt';
}

async function readGitIgnore(cwd: string): Promise<string[]> {
  try {
    const gi = await fs.readFile(path.join(cwd, '.gitignore'), 'utf8');
    const lines = gi.split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => !!l && !l.startsWith('#'));
    // Convert gitignore-ish to fast-glob ignore patterns (best-effort)
    return lines.map((p) => (p.startsWith('/') ? p.slice(1) : p));
  } catch {
    return [];
  }
}

export type DiscoverOptions = {
  cwd?: string;
  includeGlobs?: string[];
  ignoreGlobs?: string[];
  maxFileSizeBytes?: number;
};

const DEFAULT_IGNORE = [
  'node_modules/**',
  '.git/**',
  '**/*.png',
  '**/*.jpg',
  '**/*.jpeg',
  '**/*.gif',
  '**/*.pdf',
  '**/*.zip',
  '**/*.mp4',
  '**/*.mp3',
];

export async function discoverFiles(paths: string[], options: DiscoverOptions = {}): Promise<DiscoveredFile[]> {
  const cwd = options.cwd || process.cwd();
  const gitIgnores = await readGitIgnore(cwd);
  const ignore = [...DEFAULT_IGNORE, ...gitIgnores, ...(options.ignoreGlobs || [])];
  const patterns = options.includeGlobs && options.includeGlobs.length > 0 ? options.includeGlobs : ['**/*'];

  const rootSet = new Set<string>();
  for (const p of paths) rootSet.add(path.resolve(cwd, p));

  const entries: DiscoveredFile[] = [];
  for (const root of rootSet) {
    const stat = await fs.stat(root).catch(() => null);
    if (!stat) continue;
    if (stat.isFile()) {
      const ext = path.extname(root);
      if (options.maxFileSizeBytes && stat.size > options.maxFileSizeBytes) continue;
      entries.push({
        absolutePath: root,
        posixPath: toPosix(path.relative(cwd, root)),
        size: stat.size,
        ext,
        langHint: langFromExt(ext),
      });
      continue;
    }
    const found = await fg(patterns, { cwd: root, ignore, onlyFiles: true, dot: false, followSymbolicLinks: false });
    for (const rel of found) {
      const abs = path.resolve(root, rel);
      const st = await fs.stat(abs).catch(() => null);
      if (!st || !st.isFile()) continue;
      if (options.maxFileSizeBytes && st.size > options.maxFileSizeBytes) continue;
      const ext = path.extname(abs);
      entries.push({
        absolutePath: abs,
        posixPath: toPosix(path.relative(cwd, abs)),
        size: st.size,
        ext,
        langHint: langFromExt(ext),
      });
    }
  }
  return entries;
}


