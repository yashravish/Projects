const isDev = process.env.NODE_ENV === 'development';

/** Dev-only logger — no console.log in production. */
export const logger = {
  info: (...args: unknown[]) => {
    if (isDev) console.info('[AMC]', ...args);
  },
  warn: (...args: unknown[]) => {
    if (isDev) console.warn('[AMC]', ...args);
  },
  error: (...args: unknown[]) => {
    // Errors always log, even in production
    console.error('[AMC]', ...args);
  },
  perf: (label: string, fn: () => void) => {
    if (!isDev) { fn(); return; }
    const start = performance.now();
    fn();
    const elapsed = performance.now() - start;
    console.info(`[AMC:perf] ${label}: ${elapsed.toFixed(2)}ms`);
  },
};
