import path from 'node:path';
import fs from 'node:fs/promises';
import { expandHome } from '../config.js';

export type DiagramMeta = {
  id: string;
  type: 'sequence' | 'class' | 'flow';
  path: string; // file path where DSL is stored
  createdAt: number;
  params: Record<string, any>;
  sources: string[]; // chunk ids or file:line references
};

export async function saveDiagram(indexDir: string, meta: Omit<DiagramMeta, 'id' | 'path' | 'createdAt'> & { outPath?: string; dsl: string }): Promise<DiagramMeta> {
  const base = expandHome(indexDir);
  const diagramsDir = path.join(base, '..', 'diagrams');
  await fs.mkdir(diagramsDir, { recursive: true });
  const id = `${meta.type}-${Date.now()}`;
  const filePath = meta.outPath ? meta.outPath : path.join(diagramsDir, `${id}.mmd`);
  await fs.writeFile(filePath, meta.dsl, 'utf8');

  const metaPath = path.join(base, 'diagrams.json');
  let arr: DiagramMeta[] = [];
  try { arr = JSON.parse(await fs.readFile(metaPath, 'utf8')); } catch {}
  const rec: DiagramMeta = { id, type: meta.type, path: filePath, createdAt: Date.now(), params: meta.params, sources: meta.sources };
  arr.push(rec);
  await fs.mkdir(path.dirname(metaPath), { recursive: true });
  await fs.writeFile(metaPath, JSON.stringify(arr, null, 2), 'utf8');
  return rec;
}

export async function listDiagrams(indexDir: string): Promise<DiagramMeta[]> {
  const metaPath = path.join(expandHome(indexDir), 'diagrams.json');
  try { return JSON.parse(await fs.readFile(metaPath, 'utf8')); } catch { return []; }
}

export async function getDiagramById(indexDir: string, id: string): Promise<DiagramMeta | undefined> {
  const all = await listDiagrams(indexDir);
  return all.find((d) => d.id === id);
}


