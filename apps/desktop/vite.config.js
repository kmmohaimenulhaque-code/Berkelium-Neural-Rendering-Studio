import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative asset paths. The installed app opens dist/index.html straight
  // from disk (file://), where "/assets/..." points at the root of the drive
  // and the window loads blank. "./assets/..." works from disk and from the
  // dev server alike.
  base: './',
})
