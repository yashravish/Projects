import { describe, it, expect } from 'vitest';
import { validateMermaid } from '../src/validate/mermaid.js';

describe('validateMermaid', () => {
  it('accepts a simple sequence diagram', () => {
    const dsl = 'sequenceDiagram\n  A->>B: hi';
    const res = validateMermaid(dsl);
    expect(res.ok).toBe(true);
  });
  it('rejects missing header', () => {
    const dsl = 'A->>B: hi';
    const res = validateMermaid(dsl);
    expect(res.ok).toBe(false);
  });
});


