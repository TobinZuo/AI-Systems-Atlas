import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/AI-Systems-Atlas/",
  build: {
    target: "es2020",
    sourcemap: true,
  },
});
