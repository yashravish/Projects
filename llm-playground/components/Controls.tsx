"use client";
import { DEFAULT_MODEL } from "@/lib/models";

type Props = {
  model: string;
  setModel: (s: string) => void;
  temperature: number;
  setTemperature: (n: number) => void;
  top_p: number;
  setTopP: (n: number) => void;
  max_tokens: number;
  setMaxTokens: (n: number) => void;
};

// Minimal tooltips: use title attributes on labels; no popovers

export default function Controls(p: Props) {
  return (
    <div className="glass-strong rounded-xl p-5 space-y-4">
      <h2 className="text-sm">Model configuration</h2>

      <div className="flex gap-4 items-center">
        <label htmlFor="model-display" className="text-sm w-24">
          Model
        </label>
        <input
          id="model-display"
          className="glass border border-slate-600/50 p-3 rounded-lg flex-1 bg-transparent text-slate-100 font-mono text-sm text-left"
          value={p.model}
          readOnly
          disabled
        />
      </div>

      <div className="h-px bg-slate-700/40"></div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="group">
          <label htmlFor="temperature-input" className="block text-xs mb-2" title="Controls randomness. Lower = more focused; higher = more creative.">
            Temperature
          </label>
          <input
            id="temperature-input"
            className="w-full glass border border-slate-600/30 focus-neon p-3 rounded-lg transition-all font-mono text-sm text-slate-100 bg-transparent appearance-none"
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={p.temperature}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val)) {
                p.setTemperature(Math.max(0, Math.min(2, val)));
              }
            }}
          />
          <span className="text-xs text-slate-500 mt-1 block font-mono">Range: 0.0 - 2.0</span>
        </div>

        <div className="group">
          <label htmlFor="top-p-input" className="block text-xs mb-2" title="Nucleus sampling. Lower = more focused; 1.0 = full range.">
            Top P
          </label>
          <input
            id="top-p-input"
            className="w-full glass border border-slate-600/30 focus-neon p-3 rounded-lg transition-all font-mono text-sm text-slate-100 bg-transparent appearance-none"
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={p.top_p}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val)) {
                p.setTopP(Math.max(0, Math.min(1, val)));
              }
            }}
          />
          <span className="text-xs text-slate-500 mt-1 block font-mono">Range: 0.0 - 1.0</span>
        </div>

        <div className="group">
          <label htmlFor="max-tokens-input" className="block text-xs mb-2" title="Max response length (approx ~4 chars per token).">
            Max Tokens
          </label>
          <input
            id="max-tokens-input"
            className="w-full glass border border-slate-600/30 focus-neon p-3 rounded-lg transition-all font-mono text-sm text-slate-100 bg-transparent appearance-none"
            type="number"
            min={1}
            max={8192}
            value={p.max_tokens}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              if (!isNaN(val) && val > 0) {
                p.setMaxTokens(Math.min(8192, val));
              }
            }}
          />
          <span className="text-xs text-slate-500 mt-1 block font-mono">Range: 1 - 8,192</span>
        </div>
      </div>
    </div>
  );
}
