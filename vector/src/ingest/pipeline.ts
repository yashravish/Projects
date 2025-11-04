import path from 'node:path';
import fs from 'node:fs/promises';
import { discoverFiles, type DiscoverOptions } from './discover.js';
import { chunkText, type Chunk } from './chunk.js';
import { hashFile, hashString } from './hash.js';
import { type VDiagramConfig, expandHome } from '../config.js';

export type IngestStats = {
  discovered: number;
  skippedUnchanged: number;
  chunkedFiles: number;
  chunks: number;
};

type Logger = { info: (m: string) => void; debug: (m: string) => void; warn: (m: string) => void; error: (m: string) => void };

type DocIndexEntry = {
  hashHex: string;
  size: number;
  mtimeMs: number;
};

export async function runIngest(
  inputPaths: string[],
  cfg: VDiagramConfig,
  logger: Logger,
  opts: {
    includeGlobs?: string[];
    ignoreGlobs?: string[];
    maxFileSizeBytes?: number;
    targetChunkSize?: number;
    overlap?: number;
    maxChunkSize?: number;
    tag?: string;
    reset?: boolean;
  }
): Promise<{ stats: IngestStats; chunks: Chunk[] }> {
  const start = Date.now();
  const discoverOpts: DiscoverOptions = {
    includeGlobs: opts.includeGlobs,
    ignoreGlobs: [...(cfg.ignoreGlobs || []), ...(opts.ignoreGlobs || [])],
    maxFileSizeBytes: opts.maxFileSizeBytes ?? 1048576,
  };
  const files = await discoverFiles(inputPaths, discoverOpts);
  logger.info(`Discovered ${files.length} files`);

  const indexPath = path.join(expandHome(cfg.indexPath), 'doc_index.json');
  let index: Record<string, DocIndexEntry> = {};
  if (!opts.reset) {
    try {
      const raw = await fs.readFile(indexPath, 'utf8');
      index = JSON.parse(raw);
    } catch {}
  }

  const nextIndex: Record<string, DocIndexEntry> = {};
  const chunks: Chunk[] = [];
  let skippedUnchanged = 0;
  let chunkedFiles = 0;

  for (const f of files) {
    const fh = await hashFile(f.absolutePath);
    const prev = index[f.posixPath];
    const changed = !prev || prev.hashHex !== fh.hashHex || prev.size !== fh.size || prev.mtimeMs !== fh.mtimeMs;
    nextIndex[f.posixPath] = { hashHex: fh.hashHex, size: fh.size, mtimeMs: fh.mtimeMs };
    if (!changed) {
      skippedUnchanged++;
      continue;
    }
    const text = await fs.readFile(f.absolutePath, 'utf8');
    const fileChunks = chunkText(f.posixPath, text, f.langHint, {
      targetChunkSize: opts.targetChunkSize,
      overlap: opts.overlap,
      maxChunkSize: opts.maxChunkSize,
    });
    // Attach chunk-level hashes (id already includes lines; we can also hash content)
    fileChunks.forEach((ch) => {
      const chHash = hashString(ch.content);
      ch.id = `${ch.id}:${chHash.slice(0, 16)}`;
    });
    chunks.push(...fileChunks);
    chunkedFiles++;
  }

  await fs.mkdir(path.dirname(indexPath), { recursive: true });
  await fs.writeFile(indexPath, JSON.stringify(nextIndex, null, 2), 'utf8');

  const stats: IngestStats = {
    discovered: files.length,
    skippedUnchanged,
    chunkedFiles,
    chunks: chunks.length,
  };
  const dur = ((Date.now() - start) / 1000).toFixed(2);
  logger.info(`Ingest (discover→hash→chunk) completed in ${dur}s. Files: ${chunkedFiles}, chunks: ${stats.chunks}, skipped: ${skippedUnchanged}`);
  return { stats, chunks };
}


