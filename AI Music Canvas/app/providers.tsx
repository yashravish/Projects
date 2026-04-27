'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useStudioStore } from '@/store/studio-store';
import { ToastContainer } from '@/components/ui/toast';

function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000,
        refetchOnWindowFocus: false,
      },
    },
  });
}

let browserQueryClient: QueryClient | undefined;

function getQueryClient(): QueryClient {
  if (typeof window === 'undefined') {
    return makeQueryClient();
  }
  if (!browserQueryClient) {
    browserQueryClient = makeQueryClient();
  }
  return browserQueryClient;
}

function HydrationGate() {
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      useStudioStore.persist.rehydrate();
      initialized.current = true;
    }
  }, []);

  return null;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const queryClient = getQueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <HydrationGate />
      {children}
      <ToastContainer />
    </QueryClientProvider>
  );
}
