import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        primary: "#2b6cee",
        "background-dark": "#101622",
        "surface-dark": "#1c2333",
        "surface-highlight": "#282e39",
        "status-ok": "#10b981",
        "status-warn": "#f59e0b",
        "status-crit": "#ef4444"
      }
    }
  },
  plugins: []
};

export default config;
