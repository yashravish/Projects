export type Chunk = {
  id: string; // stable within file: path#startLine-endLine
  path: string; // posix
  content: string;
  startLine: number; // 1-based
  endLine: number;   // inclusive
  lang: string;
};

export type ChunkOptions = {
  targetChunkSize?: number; // characters
  overlap?: number;         // characters
  maxChunkSize?: number;    // hard limit
};

const DEFAULTS: Required<ChunkOptions> = {
  targetChunkSize: 900,
  overlap: 140,
  maxChunkSize: 1600,
};

function makeId(path: string, start: number, end: number): string {
  return `${path}#${start}-${end}`;
}

export function chunkText(
  path: string,
  content: string,
  lang: string,
  options: ChunkOptions = {}
): Chunk[] {
  const opts = { ...DEFAULTS, ...options };
  if (lang === 'md') return chunkMarkdown(path, content, lang, opts);
  if (lang === 'yaml' || lang === 'yml') return chunkByParagraphs(path, content, lang, opts);
  if (lang === 'json') return chunkByParagraphs(path, content, lang, opts);
  // Default code/text
  return chunkRecursive(path, content, lang, opts);
}

function chunkMarkdown(path: string, content: string, lang: string, opts: Required<ChunkOptions>): Chunk[] {
  // Split by headings, then apply recursive within large sections
  const lines = content.split(/\r?\n/);
  const sections: { start: number; end: number }[] = [];
  let currentStart = 1;
  for (let i = 0; i < lines.length; i++) {
    if (/^#{1,6}\s/.test(lines[i]) && i + 1 - currentStart > 0) {
      sections.push({ start: currentStart, end: i });
      currentStart = i + 1;
    }
  }
  sections.push({ start: currentStart, end: lines.length });

  const chunks: Chunk[] = [];
  for (const sec of sections) {
    const secText = lines.slice(sec.start - 1, sec.end).join('\n');
    const sub = chunkRecursive(path, secText, lang, opts, sec.start);
    chunks.push(...sub);
  }
  return chunks;
}

function chunkByParagraphs(path: string, content: string, lang: string, opts: Required<ChunkOptions>): Chunk[] {
  const lines = content.split(/\r?\n/);
  const paras: { start: number; end: number }[] = [];
  let start = 1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '') {
      if (i + 1 - start > 0) paras.push({ start, end: i });
      start = i + 2;
    }
  }
  if (start <= lines.length) paras.push({ start, end: lines.length });
  const chunks: Chunk[] = [];
  for (const p of paras) {
    const paraText = lines.slice(p.start - 1, p.end).join('\n');
    const sub = chunkRecursive(path, paraText, lang, opts, p.start);
    chunks.push(...sub);
  }
  return chunks;
}

function chunkRecursive(
  path: string,
  text: string,
  lang: string,
  opts: Required<ChunkOptions>,
  lineOffset = 1
): Chunk[] {
  const chunks: Chunk[] = [];
  const lines = text.split(/\r?\n/);
  let startLine = 1;
  let current: string[] = [];
  let currentLen = 0;

  const flush = (endLine: number) => {
    if (current.length === 0) return;
    const content = current.join('\n');
    const s = lineOffset + (startLine - 1);
    const e = lineOffset + (endLine - 1);
    chunks.push({ id: makeId(path, s, e), path, content, startLine: s, endLine: e, lang });
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const toAdd = line.length + 1; // include newline
    if (currentLen + toAdd > opts.targetChunkSize && currentLen > 0) {
      // Flush current chunk
      const endLine = i;
      flush(endLine);
      // Start next with overlap
      const overlapChars = opts.overlap;
      let backChars = 0;
      let backIdx = current.length - 1;
      const overlapLines: string[] = [];
      while (backIdx >= 0 && backChars < overlapChars) {
        const l = current[backIdx];
        overlapLines.unshift(l);
        backChars += l.length + 1;
        backIdx--;
      }
      startLine = i - overlapLines.length + 1;
      current = [...overlapLines, line];
      currentLen = current.reduce((a, b) => a + b.length + 1, 0);
    } else {
      current.push(line);
      currentLen += toAdd;
      // Hard limit guard
      if (currentLen >= opts.maxChunkSize) {
        const endLine = i + 1;
        flush(endLine);
        startLine = i + 2;
        current = [];
        currentLen = 0;
      }
    }
  }
  flush(lines.length);
  return chunks;
}


