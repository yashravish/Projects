import { HTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';

type Tone = 'neutral' | 'seal' | 'forest' | 'leaf';

interface Props extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  children: ReactNode;
}

const toneClass: Record<Tone, string> = {
  neutral: 'pill',
  seal: 'pill pill-seal',
  forest: 'pill pill-forest',
  leaf: 'pill pill-leaf',
};

export function Badge({ tone = 'neutral', className, children, ...rest }: Props) {
  return (
    <span className={cn(toneClass[tone], className)} {...rest}>
      {children}
    </span>
  );
}
