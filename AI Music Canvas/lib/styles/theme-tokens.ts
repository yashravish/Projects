import type { StyleMode } from '@/types/studio';

export interface ModeTokens {
  accent: string;
  accentRgb: string;
  secondary: string;
  tertiary: string;
  background: string;
  particleColors: string[];
}

export const MODE_TOKENS: Record<StyleMode, ModeTokens> = {
  alchemist: {
    accent: '#E8B65A',
    accentRgb: '232, 182, 90',
    secondary: '#8B6F47',
    tertiary: '#F4E8D0',
    background: '#0A0A0B',
    particleColors: ['#E8B65A', '#8B6F47', '#F4E8D0', '#C4943D', '#D4A94E'],
  },
  ambient: {
    accent: '#6FA8DC',
    accentRgb: '111, 168, 220',
    secondary: '#B19CD9',
    tertiary: '#E8E6F0',
    background: '#08090D',
    particleColors: ['#6FA8DC', '#B19CD9', '#E8E6F0', '#8BB8E8', '#9AADD6'],
  },
  trap: {
    accent: '#FF2D7A',
    accentRgb: '255, 45, 122',
    secondary: '#00F0FF',
    tertiary: '#FFFFFF',
    background: '#050507',
    particleColors: ['#FF2D7A', '#00F0FF', '#FFFFFF', '#FF5C99', '#33F2FF'],
  },
  orchestral: {
    accent: '#D4AF37',
    accentRgb: '212, 175, 55',
    secondary: '#5C1A1B',
    tertiary: '#FFFAF0',
    background: '#0B0908',
    particleColors: ['#D4AF37', '#5C1A1B', '#FFFAF0', '#B8962E', '#E6C65C'],
  },
};

/** Apply mode tokens as CSS variables on :root */
export function applyModeTokens(mode: StyleMode): void {
  const tokens = MODE_TOKENS[mode];
  const root = document.documentElement;

  root.style.setProperty('--accent', tokens.accent);
  root.style.setProperty('--accent-rgb', tokens.accentRgb);
  root.style.setProperty('--mode-secondary', tokens.secondary);
  root.style.setProperty('--mode-tertiary', tokens.tertiary);
  root.style.setProperty('--mode-bg', tokens.background);
}
