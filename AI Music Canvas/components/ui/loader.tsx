interface LoaderProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
}

const barCounts: Record<string, number> = { sm: 3, md: 5, lg: 7 };
const heights: Record<string, string> = { sm: 'h-4', md: 'h-8', lg: 'h-12' };
const gaps: Record<string, string> = { sm: 'gap-0.5', md: 'gap-1', lg: 'gap-1.5' };

export function Loader({ text, size = 'md' }: LoaderProps) {
  const count = barCounts[size];
  const height = heights[size];

  return (
    <div className="flex flex-col items-center gap-3">
      <div className={`flex items-end ${gaps[size]} ${height}`}>
        {Array.from({ length: count }, (_, i) => (
          <div
            key={i}
            className="w-1 rounded-full"
            style={{
              background: 'var(--accent)',
              height: '100%',
              animation: `waveform-pulse 1.2s ease-in-out ${i * 0.1}s infinite`,
              opacity: 0.6 + (i / count) * 0.4,
            }}
          />
        ))}
      </div>
      {text && (
        <p className="text-caption animate-pulse">{text}</p>
      )}
    </div>
  );
}
