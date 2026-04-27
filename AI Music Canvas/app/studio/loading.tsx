import { Loader } from '@/components/ui/loader';

export default function StudioLoading() {
  return (
    <div
      className="h-screen flex items-center justify-center"
      style={{ background: '#0A0A0B' }}
    >
      <Loader size="lg" text="Loading studio..." />
    </div>
  );
}
