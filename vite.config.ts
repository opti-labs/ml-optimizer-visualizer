import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // Relative asset paths so the built site works from ANY host or sub-path
  // (Netlify, Vercel, GitHub Pages project sites, plain static hosting …)
  // without hard-coding a domain. The app uses in-app state navigation (no
  // client-side routes), so no SPA-fallback config is needed.
  base: './',
  plugins: [react(), tailwindcss()],
})
