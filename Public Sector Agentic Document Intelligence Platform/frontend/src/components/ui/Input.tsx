import { forwardRef, InputHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  rubric?: string;
  hint?: ReactNode;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, Props>(function Input(
  { label, rubric, hint, error, id, className, ...rest },
  ref,
) {
  const fieldId = id ?? `field-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <label htmlFor={fieldId} className="block">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <span className="rubric">{rubric ?? label}</span>
        {hint ? <span className="datum text-2xs text-ink-60">{hint}</span> : null}
      </div>
      <input
        ref={ref}
        id={fieldId}
        className={cn('field', error && 'border-seal', className)}
        aria-invalid={!!error}
        aria-describedby={error ? `${fieldId}-error` : undefined}
        {...rest}
      />
      {error ? (
        <p
          id={`${fieldId}-error`}
          role="alert"
          className="mt-1 text-xs text-seal datum tracking-[0.04em]"
        >
          {error}
        </p>
      ) : null}
    </label>
  );
});
