import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Forces binding to all network interfaces to bypass 'localhost' Service Workers and allow external access
    allowedHosts: 'all'
  }
})