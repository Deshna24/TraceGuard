/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F19',
        panel: '#111827',
        border: '#1F2937',
        safe: '#10B981',
        warning: '#F59E0B',
        danger: '#EF4444',
        processing: '#3B82F6',
      }
    },
  },
  plugins: [],
}
