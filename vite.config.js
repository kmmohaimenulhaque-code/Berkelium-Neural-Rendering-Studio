import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: './apps/desktop/src/renderer',
  build: {
    outDir: '../../dist',
  },
  server: {
    port: 3000,
  },
});