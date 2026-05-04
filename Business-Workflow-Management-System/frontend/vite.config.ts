/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const hmrClientPort = process.env.VITE_HMR_CLIENT_PORT
  ? Number.parseInt(process.env.VITE_HMR_CLIENT_PORT, 10)
  : undefined;

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Docker Desktop + Windows bind mounts often miss native fs events.
    ...(process.env.VITE_USE_POLLING === 'true'
      ? ({ watch: { usePolling: true, interval: 300 } } as const)
      : {}),
    hmr:
      hmrClientPort != null && !Number.isNaN(hmrClientPort)
        ? { clientPort: hmrClientPort }
        : undefined,
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    css: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
});
