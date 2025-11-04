"use client";
import { useState } from "react";

export default function TrainPage() {
  const [text, setText] = useState("Paste a small corpus here...");
  const [losses, setLosses] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setError(null);
    setLosses(null);

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data?.error || "Training failed");
        return;
      }

      setLosses(data.losses ?? null);
    } catch (e) {
      setError("Failed to start training");
      console.error("Training error:", e);
    }
  }

  return (
    <main className="p-6 max-w-2xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">Toy Training Demo</h1>
      <textarea
        className="w-full border p-3 rounded h-40"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        onClick={start}
        className="bg-black text-white px-4 py-2 rounded hover:bg-gray-800 transition-colors"
      >
        Start
      </button>
      {error && (
        <div className="text-red-700 bg-red-50 border border-red-200 p-2 rounded">
          {error}
        </div>
      )}
      {losses && (
        <div className="border rounded p-3">
          <h2 className="font-medium mb-2">Loss per Epoch</h2>
          <ul className="list-disc pl-5">
            {losses.map((v, i) => (
              <li key={i}>
                Epoch {i}: {v.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
