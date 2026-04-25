import { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

interface Props extends HTMLAttributes<HTMLDivElement> {
  rows?: number;
}

export function Skeleton({ rows = 4, className, ...rest }: Props) {
  return (
    <div className={cn('space-y-2', className)} aria-hidden {...rest}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-4 animate-ink-blink"
          style={{
            width: `${85 - (i % 4) * 12}%`,
            background:
              'linear-gradient(90deg, rgba(14,14,17,0.06), rgba(14,14,17,0.14), rgba(14,14,17,0.06))',
          }}
        />
      ))}
    </div>
  );
}
