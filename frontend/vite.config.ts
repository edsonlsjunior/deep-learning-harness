import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// O proxy evita CORS e permite streaming SSE do /api/chat
// apresentacao.html = slides "O que é um harness" (em dev: http://localhost:5173/apresentacao)
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // caminhos relativos sao resolvidos pelo Vite a partir do root (frontend/)
      input: { main: "index.html", apresentacao: "apresentacao.html" },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
