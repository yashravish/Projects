'use client';

import dynamic from 'next/dynamic';
import { Loader } from '@/components/ui/loader';

const StudioClient = dynamic(() => import('./studio-client'), {
  ssr: false,
  loading: () => (
    <div className="h-screen flex items-center justify-center" style={{ background: '#0A0A0B' }}>
      <Loader size="lg" text="Loading studio..." />
    </div>
  ),
});

export default function StudioPage() {
  return <StudioClient />;
}
