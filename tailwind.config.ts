import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: "#08111f",
        command: "#0b1728",
        cyanline: "#22d3ee",
        electric: "#38bdf8",
        warning: "#f59e0b",
        danger: "#ef4444"
      },
      boxShadow: {
        glow: "0 0 32px rgba(34, 211, 238, 0.18)"
      }
    }
  },
  plugins: []
};

export default config;
