/**
 * Raga definitions with allowed swaras
 */

import type { Swara } from "../types";

/**
 * Raga definition with arohanam (ascending) and avarohanam (descending) swaras
 */
export interface RagaDefinition {
  name: string;
  arohanam: Swara[];
  avarohanam: Swara[];
  /** All allowed swaras (union of arohanam and avarohanam) */
  allowedSwaras: Swara[];
  /** Primary/important swaras */
  vadi?: Swara;
  samvadi?: Swara;
}

/**
 * Common Carnatic ragas (L0 set)
 */
export const RAGAS: Record<string, RagaDefinition> = {
  Mohanam: {
    name: "Mohanam",
    arohanam: ["S", "R2", "G3", "P", "D2"],
    avarohanam: ["D2", "P", "G3", "R2", "S"],
    allowedSwaras: ["S", "R2", "G3", "P", "D2"],
    vadi: "G3",
    samvadi: "D2",
  },

  Sankarabharanam: {
    name: "Sankarabharanam",
    arohanam: ["S", "R2", "G3", "M1", "P", "D2", "N3"],
    avarohanam: ["N3", "D2", "P", "M1", "G3", "R2", "S"],
    allowedSwaras: ["S", "R2", "G3", "M1", "P", "D2", "N3"],
    vadi: "D2",
    samvadi: "G3",
  },

  Kalyani: {
    name: "Kalyani",
    arohanam: ["S", "R2", "G3", "M2", "P", "D2", "N3"],
    avarohanam: ["N3", "D2", "P", "M2", "G3", "R2", "S"],
    allowedSwaras: ["S", "R2", "G3", "M2", "P", "D2", "N3"],
    vadi: "G3",
    samvadi: "N3",
  },

  Mayamalavagowla: {
    name: "Mayamalavagowla",
    arohanam: ["S", "R1", "G3", "M1", "P", "D1", "N3"],
    avarohanam: ["N3", "D1", "P", "M1", "G3", "R1", "S"],
    allowedSwaras: ["S", "R1", "G3", "M1", "P", "D1", "N3"],
    vadi: "G3",
    samvadi: "N3",
  },

  Kharaharapriya: {
    name: "Kharaharapriya",
    arohanam: ["S", "R2", "G2", "M1", "P", "D2", "N2"],
    avarohanam: ["N2", "D2", "P", "M1", "G2", "R2", "S"],
    allowedSwaras: ["S", "R2", "G2", "M1", "P", "D2", "N2"],
    vadi: "G2",
    samvadi: "D2",
  },

  Hamsadhwani: {
    name: "Hamsadhwani",
    arohanam: ["S", "R2", "G3", "P", "N3"],
    avarohanam: ["N3", "P", "G3", "R2", "S"],
    allowedSwaras: ["S", "R2", "G3", "P", "N3"],
    vadi: "G3",
    samvadi: "N3",
  },

  Bhairavi: {
    name: "Bhairavi",
    arohanam: ["S", "R2", "G2", "M1", "P", "D2", "N2"],
    avarohanam: ["N2", "D1", "P", "M1", "G2", "R1", "S"],
    allowedSwaras: ["S", "R1", "R2", "G2", "M1", "P", "D1", "D2", "N2"],
    vadi: "M1",
    samvadi: "S",
  },

  Shanmukhapriya: {
    name: "Shanmukhapriya",
    arohanam: ["S", "R1", "G3", "M2", "P", "D1", "N3"],
    avarohanam: ["N3", "D1", "P", "M2", "G3", "R1", "S"],
    allowedSwaras: ["S", "R1", "G3", "M2", "P", "D1", "N3"],
  },

  Thodi: {
    name: "Thodi",
    arohanam: ["S", "R1", "G2", "M1", "P", "D1", "N2"],
    avarohanam: ["N2", "D1", "P", "M1", "G2", "R1", "S"],
    allowedSwaras: ["S", "R1", "G2", "M1", "P", "D1", "N2"],
    vadi: "D1",
    samvadi: "G2",
  },

  Abhogi: {
    name: "Abhogi",
    arohanam: ["S", "R2", "G2", "M1", "D2"],
    avarohanam: ["D2", "M1", "G2", "R2", "S"],
    allowedSwaras: ["S", "R2", "G2", "M1", "D2"],
  },
};

/**
 * Get raga definition by name (case-insensitive)
 */
export function getRaga(name: string): RagaDefinition | null {
  const normalized = name.toLowerCase().replace(/\s+/g, "");

  for (const [key, raga] of Object.entries(RAGAS)) {
    if (key.toLowerCase() === normalized) {
      return raga;
    }
  }

  return null;
}

/**
 * Get list of all available raga names
 */
export function getAvailableRagas(): string[] {
  return Object.keys(RAGAS);
}

/**
 * Check if a swara is allowed in a raga
 */
export function isSwaraAllowedInRaga(swara: Swara, ragaName: string): boolean {
  const raga = getRaga(ragaName);
  if (!raga) {
    return true; // Unknown raga, allow all
  }

  return raga.allowedSwaras.includes(swara);
}
