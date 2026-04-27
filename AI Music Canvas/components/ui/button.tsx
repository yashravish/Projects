import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'accent';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: ReactNode;
  isLoading?: boolean;
}

const variantStyles: Record<ButtonVariant, { base: string; hover: string; active: string }> = {
  primary: {
    base: 'background: var(--accent); color: #0A0A0B; border: 1px solid transparent;',
    hover: 'filter: brightness(1.1); transform: translateY(-1px);',
    active: 'filter: brightness(0.95); transform: translateY(0);',
  },
  secondary: {
    base: 'background: rgba(255,255,255,0.06); color: var(--foreground); border: 1px solid rgba(255,255,255,0.1);',
    hover: 'background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.15);',
    active: 'background: rgba(255,255,255,0.08);',
  },
  ghost: {
    base: 'background: transparent; color: rgba(255,255,255,0.6); border: 1px solid transparent;',
    hover: 'background: rgba(255,255,255,0.04); color: var(--foreground);',
    active: 'background: rgba(255,255,255,0.06);',
  },
  accent: {
    base: 'background: rgba(var(--accent-rgb),0.12); color: var(--accent); border: 1px solid rgba(var(--accent-rgb),0.2);',
    hover: 'background: rgba(var(--accent-rgb),0.2); border-color: rgba(var(--accent-rgb),0.3);',
    active: 'background: rgba(var(--accent-rgb),0.15);',
  },
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs gap-1.5',
  md: 'px-4 py-2 text-sm gap-2',
  lg: 'px-6 py-3 text-base gap-2.5',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'secondary', size = 'md', icon, isLoading, children, disabled, className = '', style, ...props }, ref) => {
    const v = variantStyles[variant];

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`
          inline-flex items-center justify-center font-medium
          rounded-[var(--radius-button)]
          transition-all duration-200
          ${sizeStyles[size]}
          ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
          ${className}
        `}
        style={{
          ...Object.fromEntries(v.base.split(';').filter(Boolean).map(s => {
            const [k, ...rest] = s.split(':');
            return [k.trim().replace(/-([a-z])/g, (_, c: string) => c.toUpperCase()), rest.join(':').trim()];
          })),
          ...style,
        }}
        onMouseEnter={(e) => {
          if (disabled) return;
          const el = e.currentTarget;
          v.hover.split(';').filter(Boolean).forEach(s => {
            const [k, ...rest] = s.split(':');
            el.style.setProperty(k.trim(), rest.join(':').trim());
          });
        }}
        onMouseLeave={(e) => {
          const el = e.currentTarget;
          v.base.split(';').filter(Boolean).forEach(s => {
            const [k, ...rest] = s.split(':');
            el.style.setProperty(k.trim(), rest.join(':').trim());
          });
        }}
        onMouseDown={(e) => {
          if (disabled) return;
          const el = e.currentTarget;
          v.active.split(';').filter(Boolean).forEach(s => {
            const [k, ...rest] = s.split(':');
            el.style.setProperty(k.trim(), rest.join(':').trim());
          });
        }}
        onMouseUp={(e) => {
          if (disabled) return;
          const el = e.currentTarget;
          v.hover.split(';').filter(Boolean).forEach(s => {
            const [k, ...rest] = s.split(':');
            el.style.setProperty(k.trim(), rest.join(':').trim());
          });
        }}
        {...props}
      >
        {isLoading ? (
          <span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : icon ? (
          <span className="flex-shrink-0">{icon}</span>
        ) : null}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
