import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, /api/* is forwarded to the FastAPI server so there's no CORS hassle.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
