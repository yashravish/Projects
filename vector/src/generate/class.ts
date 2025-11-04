import type { QueryResult } from '../store/types.js';

function pickTopEntities(results: QueryResult[]): string[] {
  const counts = new Map<string, number>();
  for (const r of results) {
    const base = r.record.path.split('/').pop() || 'Entity';
    const name = base.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_]/g, '_');
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([n]) => n);
}

export function generateClassMermaid(topic: string, results: QueryResult[]): { dsl: string; provenance: string[] } {
  const entities = pickTopEntities(results);
  const lines: string[] = [];
  lines.push('classDiagram');
  for (const e of entities) {
    lines.push(`  class ${e} {`);
    lines.push('  }');
  }
  for (let i = 0; i + 1 < entities.length; i++) {
    lines.push(`  ${entities[i]} -- ${entities[i + 1]}`);
  }
  const provenance = results.slice(0, 10).map((r) => `${r.record.path}:${r.record.startLine}-${r.record.endLine}`);
  return { dsl: lines.join('\n'), provenance };
}


