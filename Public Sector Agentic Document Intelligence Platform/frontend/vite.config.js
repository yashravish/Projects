/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    server: {
        host: '0.0.0.0',
        port: 5173,
        strictPort: true,
        watch: {
            usePolling: true,
            interval: 250,
        },
        proxy: {
            '/api': {
                target: process.env.VITE_API_PROXY_TARGET || 'http://api:8000',
                changeOrigin: true,
            },
            '/health': {
                target: process.env.VITE_API_PROXY_TARGET || 'http://api:8000',
                changeOrigin: true,
            },
        },
    },
    build: {
        target: 'es2022',
        sourcemap: true,
        chunkSizeWarningLimit: 800,
    },
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./src/test/setup.ts'],
    },
});
