/**
 * 深色主题配置 (Spotify Style)
 */
export const theme = {
  colors: {
    primary: {
      50: '#0a0a0a',
      100: '#121212',
      200: '#181818',
      300: '#242424',
      400: '#333333',
      500: '#535353',
      600: '#b3b3b3',
      700: '#e0e0e0',
      800: '#f6f6f6',
      900: '#ffffff',
      accent: '#1db954', // Spotify Green
    },
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
      muted: '#a7a7a7',
    },
    background: {
      main: '#000000',
      card: '#121212',
      hover: '#1a1a1a',
      elevated: '#242424',
    },
    border: {
      default: '#2a2a2a',
      focus: '#404040',
    },
  },
  gradients: {
    primary: 'linear-gradient(135deg, #242424 0%, #121212 100%)',
    accent: 'linear-gradient(135deg, #1db954 0%, #179342 100%)',
    background: '#000000',
    hero: 'linear-gradient(180deg, rgba(83,83,83,0.3) 0%, rgba(18,18,18,1) 100%)',
  },
  spacing: {
    xs: '0.5rem',
    sm: '1rem',
    md: '1.5rem',
    lg: '2rem',
    xl: '3rem',
  },
  borderRadius: {
    sm: '0.4rem',
    md: '0.6rem',
    lg: '1rem',
    full: '9999px',
  },
  shadows: {
    sm: '0 4px 12px rgba(0, 0, 0, 0.5)',
    md: '0 8px 24px rgba(0, 0, 0, 0.5)',
    lg: '0 16px 32px rgba(0, 0, 0, 0.6)',
  },
  layout: {
    sidebarWidth: 260,
    contentMaxWidth: 1280,
    maxWidth: 1600,
  },
};

