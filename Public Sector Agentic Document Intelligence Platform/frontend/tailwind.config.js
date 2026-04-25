/**
 * Federal Dossier — design tokens.
 *
 * The aesthetic is editorial / archival: ink on parchment, hairline rules as
 * primary structure, restrained motion. The palette is deliberately narrow
 * (one ink, two papers, three accents) so callsites are forced into intent.
 */
var config = {
    content: ['./index.html', './src/**/*.{ts,tsx}'],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                ink: {
                    DEFAULT: '#0E0E11',
                    80: 'rgba(14, 14, 17, 0.80)',
                    60: 'rgba(14, 14, 17, 0.60)',
                    40: 'rgba(14, 14, 17, 0.40)',
                    20: 'rgba(14, 14, 17, 0.20)',
                    10: 'rgba(14, 14, 17, 0.10)',
                    5: 'rgba(14, 14, 17, 0.05)',
                },
                paper: {
                    DEFAULT: '#F2EEE3',
                    deep: '#E8E2D2',
                    well: '#DDD6C2',
                },
                seal: {
                    DEFAULT: '#7C1D1D',
                    deep: '#5A1414',
                },
                forest: {
                    DEFAULT: '#2D3B2A',
                    soft: '#4F614B',
                },
                leaf: {
                    DEFAULT: '#9A7A2B',
                    deep: '#6E561A',
                },
                muted: '#5C544A',
            },
            fontFamily: {
                display: ['"Newsreader Variable"', 'Newsreader', 'Georgia', 'serif'],
                sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
                mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
            },
            fontSize: {
                '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.04em' }],
                xs: ['0.75rem', { lineHeight: '1.1rem', letterSpacing: '0.02em' }],
                sm: ['0.8125rem', { lineHeight: '1.25rem' }],
                base: ['0.9375rem', { lineHeight: '1.65rem' }],
                lg: ['1.0625rem', { lineHeight: '1.75rem' }],
                xl: ['1.25rem', { lineHeight: '1.85rem' }],
                '2xl': ['1.5rem', { lineHeight: '1.95rem', letterSpacing: '-0.005em' }],
                '3xl': ['1.875rem', { lineHeight: '2.15rem', letterSpacing: '-0.01em' }],
                '4xl': ['2.5rem', { lineHeight: '2.7rem', letterSpacing: '-0.015em' }],
                '5xl': ['3.5rem', { lineHeight: '3.7rem', letterSpacing: '-0.02em' }],
            },
            letterSpacing: {
                rubric: '0.18em',
            },
            borderColor: {
                DEFAULT: 'rgba(14, 14, 17, 0.18)',
                rule: 'rgba(14, 14, 17, 0.85)',
            },
            borderWidth: {
                hair: '0.5px',
            },
            maxWidth: {
                prose: '68ch',
                column: '74rem',
            },
            animation: {
                'rise-in': 'riseIn 600ms cubic-bezier(0.2, 0.7, 0.1, 1) both',
                'ink-blink': 'inkBlink 1.6s ease-in-out infinite',
            },
            keyframes: {
                riseIn: {
                    '0%': { opacity: '0', transform: 'translateY(8px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                inkBlink: {
                    '0%, 100%': { opacity: '0.45' },
                    '50%': { opacity: '0.95' },
                },
            },
            backgroundImage: {
                'paper-grain': "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.05 0 0 0 0 0.05 0 0 0 0 0.06 0 0 0 0.06 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
            },
        },
    },
    plugins: [],
};
export default config;
