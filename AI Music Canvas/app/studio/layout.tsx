import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Studio — AI Music Canvas',
  description: 'Drop an audio file and watch it transform into living visual art.',
};

export default function StudioLayout({ children }: { children: React.ReactNode }) {
  return children;
}
