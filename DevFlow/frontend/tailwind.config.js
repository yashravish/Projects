/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'DM Sans'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      colors: {
        brand: {
          50: "#f0f4ff",
          100: "#e0e9ff",
          500: "#3b4cca",
          600: "#2f3eb0",
          900: "#0f1b4a",
        },
        surface: {
          900: "#0b0f1a",
          800: "#111827",
          700: "#1f2937",
        },
      },
    },
  },
  plugins: [],
};
