'use client';

import { useStudioStore } from '@/store/studio-store';
import { Slider } from '@/components/ui/slider';
import { RotateCcw } from 'lucide-react';
import type { ControlValues } from '@/types/studio';

const SLIDER_CONFIGS: {
  key: keyof ControlValues;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}[] = [
  { key: 'intensity', label: 'Intensity', min: 0, max: 100, step: 1, unit: '%' },
  { key: 'particleCount', label: 'Particles', min: 50, max: 2000, step: 10, unit: '' },
  { key: 'blur', label: 'Blur', min: 0, max: 20, step: 0.5, unit: 'px' },
  { key: 'glow', label: 'Glow', min: 0, max: 100, step: 1, unit: '%' },
  { key: 'waveformThickness', label: 'Waveform', min: 1, max: 8, step: 0.5, unit: 'px' },
  { key: 'backgroundMotion', label: 'Bg Motion', min: 0, max: 100, step: 1, unit: '%' },
];

export function ControlPanel() {
  const controls = useStudioStore((s) => s.controls);
  const setControl = useStudioStore((s) => s.setControl);
  const resetControls = useStudioStore((s) => s.resetControls);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-caption">Controls</p>
        <button
          onClick={resetControls}
          className="flex items-center gap-1 text-[10px] uppercase tracking-wider transition-colors duration-200 cursor-pointer"
          style={{ color: 'rgba(255,255,255,0.3)' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.3)'; }}
          aria-label="Reset all controls to defaults"
        >
          <RotateCcw size={10} strokeWidth={1.5} />
          Reset
        </button>
      </div>

      <div className="space-y-3">
        {SLIDER_CONFIGS.map((config) => (
          <Slider
            key={config.key}
            label={config.label}
            value={controls[config.key]}
            min={config.min}
            max={config.max}
            step={config.step}
            unit={config.unit}
            onChange={(v) => setControl(config.key, v)}
          />
        ))}
      </div>
    </div>
  );
}
