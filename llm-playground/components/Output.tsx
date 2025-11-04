"use client";
import { useState } from "react";

type Props = { text: string; isStreaming?: boolean };

export default function Output({ text, isStreaming = false }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const isEmpty = !text.trim();

  return (
    <div className="glass-strong rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <label className="block text-sm">Output</label>
        {!isEmpty && (
          <button onClick={handleCopy} className="text-xs text-slate-300 hover:text-white">
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>
      <pre
        className={`glass border rounded-lg p-4 whitespace-pre-wrap min-h-40 font-mono text-sm ${
          isEmpty ? "border-slate-700/30 text-slate-500" : "border-slate-600/30 text-slate-100"
        }`}
        role="status"
        aria-live="polite"
      >
        {isEmpty ? "…" : text}
      </pre>
    </div>
  );
}
