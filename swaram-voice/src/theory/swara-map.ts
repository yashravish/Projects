/**
 * Swara mapping from cents to Carnatic swara names
 */

import type { Cents, Swara } from "../types";

/**
 * Canonical cents values from Sa for each swara
 */
export const SWARA_CENTS: Record<Swara, Cents> = {
  S: 0,
  R1: 90,
  R2: 204,
  G2: 294, // Corresponds to G2 (antara gandhara)
  G3: 386,
  M1: 498,
  M2: 590,
  P: 702,
  D1: 792,
  D2: 906,
  N2: 996,
  N3: 1088,
};

/**
 * All swaras in ascending order
 */
export const ALL_SWARAS: Swara[] = [
  "S",
  "R1",
  "R2",
  "G2",
  "G3",
  "M1",
  "M2",
  "P",
  "D1",
  "D2",
  "N2",
  "N3",
];

/**
 * Map cents value to nearest swara
 * @param cents Cents from Sa (tonic)
 * @param allowedSwaras Optional list of allowed swaras (for raga constraints)
 * @param snapTolerance Maximum distance in cents to snap to a swara
 * @returns Object with swara, exact cents, and confidence
 */
export function centsToSwara(
  cents: Cents,
  allowedSwaras: Swara[] = ALL_SWARAS,
  snapTolerance: number = 50
): { swara: Swara; cents: Cents; confidence: number } | null {
  // Normalize cents to 0-1200 range (one octave)
  const normalizedCents = ((cents % 1200) + 1200) % 1200;

  let minDistance = Infinity;
  let closestSwara: Swara | null = null;

  for (const swara of allowedSwaras) {
    const swaraCents = SWARA_CENTS[swara];
    
    // Consider both direct distance and octave-wrapped distance
    const distance = Math.min(
      Math.abs(normalizedCents - swaraCents),
      Math.abs(normalizedCents - swaraCents - 1200),
      Math.abs(normalizedCents - swaraCents + 1200)
    );

    if (distance < minDistance) {
      minDistance = distance;
      closestSwara = swara;
    }
  }

  if (closestSwara === null || minDistance > snapTolerance) {
    return null;
  }

  // Calculate confidence based on distance (closer = higher confidence)
  const confidence = Math.max(0, 1 - minDistance / snapTolerance);

  return {
    swara: closestSwara,
    cents: normalizedCents,
    confidence,
  };
}

/**
 * Get swara name with octave information
 */
export function getSwaraWithOctave(cents: Cents, baseSa: Cents = 0): string {
  const octaveNumber = Math.floor((cents - baseSa) / 1200);
  const normalizedCents = ((cents % 1200) + 1200) % 1200;
  
  const result = centsToSwara(normalizedCents);
  if (!result) {
    return "?";
  }

  const octaveMarker = octaveNumber > 0 ? "'" : octaveNumber < 0 ? "," : "";
  return result.swara + octaveMarker;
}

/**
 * Check if a cents value is close to a swara
 */
export function isNearSwara(cents: Cents, swara: Swara, tolerance: number = 25): boolean {
  const normalizedCents = ((cents % 1200) + 1200) % 1200;
  const swaraCents = SWARA_CENTS[swara];
  
  const distance = Math.min(
    Math.abs(normalizedCents - swaraCents),
    Math.abs(normalizedCents - swaraCents - 1200),
    Math.abs(normalizedCents - swaraCents + 1200)
  );

  return distance <= tolerance;
}

/**
 * Get cents value for a swara
 */
export function getSwaraCents(swara: Swara): Cents {
  return SWARA_CENTS[swara];
}

/**
 * Calculate interval between two swaras in cents
 */
export function swaraInterval(swara1: Swara, swara2: Swara): Cents {
  const cents1 = SWARA_CENTS[swara1];
  const cents2 = SWARA_CENTS[swara2];
  return cents2 - cents1;
}
