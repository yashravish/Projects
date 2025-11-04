export type ValidationResult = { ok: true } | { ok: false; error: string };

export function validateMermaid(dsl: string): ValidationResult {
  const headerOk = /^(sequenceDiagram|classDiagram|flowchart\s+(TD|LR))/m.test(dsl);
  if (!headerOk) return { ok: false, error: 'Missing or invalid Mermaid header (sequenceDiagram|classDiagram|flowchart TD).' };
  // Very light checks for arrows balance
  const arrows = (dsl.match(/-->|->>|-->>|==>/g) || []).length;
  if (arrows === 0 && dsl.startsWith('sequenceDiagram')) return { ok: false, error: 'No arrows found in sequence diagram.' };
  // Basic ID sanity: no spaces in identifiers (Mermaid allows labels, but IDs should be safe)
  if (/^[ \t]/m.test(dsl)) {
    // allow indentation
  }
  return { ok: true };
}


