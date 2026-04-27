'use client';

import * as SliderPrimitive from '@radix-ui/react-slider';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
  formatValue?: (value: number) => string;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit = '',
  onChange,
  formatValue,
}: SliderProps) {
  const displayValue = formatValue ? formatValue(value) : `${Math.round(value)}${unit}`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-caption">{label}</label>
        <span
          className="font-mono text-xs tabular-nums"
          style={{ color: 'var(--accent)', minWidth: '3ch', textAlign: 'right' }}
        >
          {displayValue}
        </span>
      </div>
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-5 cursor-pointer"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
        aria-label={label}
        aria-valuetext={displayValue}
      >
        <SliderPrimitive.Track className="relative h-1 w-full rounded-full" style={{ background: 'rgba(255,255,255,0.08)' }}>
          <SliderPrimitive.Range className="absolute h-full rounded-full" style={{ background: 'var(--accent)' }} />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block w-3.5 h-3.5 rounded-full border-2 transition-transform duration-150 hover:scale-125"
          style={{
            background: 'var(--foreground)',
            borderColor: 'var(--accent)',
          }}
        />
      </SliderPrimitive.Root>
    </div>
  );
}
