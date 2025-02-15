/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#382D76',
          50: '#E8E6F0',
          100: '#D1CCE1',
          200: '#A399C3',
          300: '#7566A5',
          400: '#4C3F8A',
          500: '#382D76',
          600: '#2D245F',
          700: '#221B47',
          800: '#16122F',
          900: '#0B0918'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif']
      }
    }
  },
  plugins: []
};