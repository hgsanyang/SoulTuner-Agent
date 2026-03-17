'use client';

import { useEffect, useState } from 'react';

/**
 * Subscribe to a media query and keep the match result in sync with resize events.
 * Falls back to `false` during SSR and hydrates once mounted.
 */
export function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return;
    }

    const mediaQuery = window.matchMedia(query);
    const updateMatch = () => setMatches(mediaQuery.matches);

    updateMatch();
    mediaQuery.addEventListener('change', updateMatch);

    return () => {
      mediaQuery.removeEventListener('change', updateMatch);
    };
  }, [query]);

  return matches;
}


