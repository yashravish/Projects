"use client";
import { useEffect, useRef, useState } from "react";

type Frame = [number, string];

type Props = {
  frames: Frame[]; // ms, token
  finalText: string;
};

export default function ReplayPlayer({ frames, finalText }: Props) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [cursor, setCursor] = useState(0); // next frame index to render
  const [text, setText] = useState("");
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0); // playback epoch (performance.now())

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  function play() {
    if (isPlaying) return;
    setIsPlaying(true);
    // Resume from current cursor position (use previous frame time as the offset)
    const resumeAtMs = cursor > 0 ? frames[Math.min(cursor - 1, frames.length - 1)][0] : 0;
    startTimeRef.current = performance.now() - resumeAtMs / speed;
    rafRef.current = requestAnimationFrame(loop);
  }

  function pause() {
    setIsPlaying(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }

  function loop() {
    if (!isPlaying) return;
    const elapsed = (performance.now() - startTimeRef.current) * speed; // ms since (virtual) start
    let nextIndex = cursor;
    let buffer = text;

    while (nextIndex < frames.length && frames[nextIndex][0] <= elapsed) {
      buffer += frames[nextIndex][1];
      nextIndex++;
    }

    if (nextIndex !== cursor) {
      setCursor(nextIndex);
      setText(buffer);
    }

    if (nextIndex >= frames.length) {
      setIsPlaying(false);
      setText(finalText);
      return;
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  function reset() {
    pause();
    setCursor(0);
    setText("");
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {!isPlaying ? (
          <button onClick={play} className="px-3 py-1.5 glass border border-slate-600/40 rounded text-sm">Play</button>
        ) : (
          <button onClick={pause} className="px-3 py-1.5 glass border border-slate-600/40 rounded text-sm">Pause</button>
        )}
        <button onClick={reset} className="px-3 py-1.5 glass border border-slate-600/40 rounded text-sm">Reset</button>
        <label className="text-xs text-slate-400">Speed</label>
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="glass border border-slate-600/40 rounded px-2 py-1 text-xs"
        >
          <option value={0.5}>0.5x</option>
          <option value={1}>1x</option>
          <option value={2}>2x</option>
          <option value={4}>4x</option>
        </select>
      </div>
      <pre className="glass border border-slate-600/40 rounded p-4 whitespace-pre-wrap min-h-40 text-sm">{text}</pre>
    </div>
  );
}


