'use client';

import { useEffect, useState, useCallback } from 'react';

interface MediaQueryState {
  prefersReducedMotion: boolean;
  isTouch: boolean;
  isMobile: boolean;
  isTablet: boolean;
}

export function useMediaQuery(): MediaQueryState {
  const [state, setState] = useState<MediaQueryState>({
    prefersReducedMotion: false,
    isTouch: false,
    isMobile: false,
    isTablet: false,
  });

  const update = useCallback(() => {
    setState({
      prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
      isTouch: window.matchMedia('(pointer: coarse)').matches,
      isMobile: window.matchMedia('(max-width: 640px)').matches,
      isTablet: window.matchMedia('(max-width: 1024px)').matches,
    });
  }, []);

  useEffect(() => {
    update();

    const queries = [
      window.matchMedia('(prefers-reduced-motion: reduce)'),
      window.matchMedia('(pointer: coarse)'),
      window.matchMedia('(max-width: 640px)'),
      window.matchMedia('(max-width: 1024px)'),
    ];

    const handler = () => update();

    for (const mql of queries) {
      mql.addEventListener('change', handler);
    }

    return () => {
      for (const mql of queries) {
        mql.removeEventListener('change', handler);
      }
    };
  }, [update]);

  return state;
}
