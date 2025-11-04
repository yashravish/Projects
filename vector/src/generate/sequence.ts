import type { QueryResult } from '../store/types.js';

function extractParticipants(results: QueryResult[]): string[] {
  const names = new Map<string, number>();
  for (const r of results) {
    const parts = r.record.path.split('/');
    const mod = parts.length > 1 ? parts[0] : parts[0];
    const key = mod.replace(/[^A-Za-z0-9_]/g, '_');
    names.set(key, (names.get(key) || 0) + 1);
  }
  return Array.from(names.keys()).slice(0, 8);
}

export function generateSequenceMermaid(question: string, results: QueryResult[]): { dsl: string; provenance: string[] } {
  const participants = extractParticipants(results);
  if (participants.length === 0) {
    // Fallback when no results
    const lines = ['sequenceDiagram', '  participant User', '  participant System', '  User->>System: request', '  System-->>User: response'];
    return { dsl: lines.join('\n'), provenance: [] };
  }
  const lines: string[] = [];
  lines.push('sequenceDiagram');
  participants.forEach((p) => lines.push(`  participant ${p}`));
  // Ensure at least one arrow
  const arrowCount = Math.max(1, Math.min(results.length, participants.length * 2));
  for (let i = 0; i < arrowCount; i++) {
    const from = participants[i % participants.length];
    const to = participants[(i + 1) % participants.length];
    const label = question.slice(0, 24).replace(/:/g, ' ') || 'action';
    lines.push(`  ${from}->>${to}: ${label}`);
  }
  const provenance = results.slice(0, 8).map((r) => `${r.record.path}:${r.record.startLine}-${r.record.endLine}`);
  return { dsl: lines.join('\n'), provenance };
}


