import { HTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'title'> {
  rubric?: string;
  title?: ReactNode;
  meta?: ReactNode;
  variant?: 'plain' | 'deep';
  children: ReactNode;
}

export function Card({
  rubric,
  title,
  meta,
  variant = 'plain',
  className,
  children,
  ...rest
}: Props) {
  return (
    <section
      className={cn(variant === 'deep' ? 'panel-deep' : 'panel', 'p-6', className)}
      {...rest}
    >
      {(rubric || title || meta) && (
        <header className="mb-5">
          {rubric ? <p className="rubric">{rubric}</p> : null}
          <div className="flex items-baseline justify-between gap-4">
            {title ? (
              <h3 className="display text-2xl mt-1">{title}</h3>
            ) : (
              <span />
            )}
            {meta ? <div className="datum text-xs text-ink-60">{meta}</div> : null}
          </div>
          <hr className="rule-soft mt-4" />
        </header>
      )}
      {children}
    </section>
  );
}
