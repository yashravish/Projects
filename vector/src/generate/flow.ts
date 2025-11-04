import type { QueryResult } from '../store/types.js';

function extractSteps(results: QueryResult[]): string[] {
  const steps = new Set<string>();
  for (const r of results) {
    const content = r.record.content;
    const matches = content.match(/name:\s*['\"]?([A-Za-z0-9 _-]{3,50})/g) || [];
    for (const m of matches) {
      const name = m.split(':')[1].trim().replace(/["']/g, '').replace(/[^A-Za-z0-9_ -]/g, '').slice(0, 32);
      if (name) steps.add(name);
    }
  }
  const arr = Array.from(steps);
  return arr.slice(0, 12);
}

export function generateFlowMermaid(question: string, results: QueryResult[]): { dsl: string; provenance: string[] } {
  const steps = extractSteps(results);
  const lines: string[] = [];
  lines.push('flowchart TD');
  for (let i = 0; i < steps.length; i++) {
    const id = `S${i}`;
    lines.push(`  ${id}[${steps[i]}]`);
    if (i > 0) lines.push(`  S${i - 1} --> ${id}`);
  }
  if (steps.length === 0) {
    lines.push(`  A[${question.slice(0, 24)}] --> B[Review configs]`);
  }
  const provenance = results.slice(0, 8).map((r) => `${r.record.path}:${r.record.startLine}-${r.record.endLine}`);
  return { dsl: lines.join('\n'), provenance };
}


