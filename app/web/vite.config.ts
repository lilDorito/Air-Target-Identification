import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// in prod remember to set VITE_API_URL
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
