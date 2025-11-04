"use client";
import { useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import Controls from "@/components/Controls";
import Output from "@/components/Output";
import RunsTable from "@/components/RunsTable";
import { estimateCostUSD, roughTokenEstimate } from "@/lib/pricing";
import { DEFAULT_MODEL } from "@/lib/models";
import { LIMITS } from "@/lib/constants";

export default function Page() {
  const { data: session } = useSession();
  const [model, setModel] = useState<string>(DEFAULT_MODEL);
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [top_p, setTopP] = useState(1);
  const [max_tokens, setMaxTokens] = useState(256);
  const [output, setOutput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  // Share replay UI removed

  const runBtnRef = useRef<HTMLButtonElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Keyboard shortcut: Cmd/Ctrl + Enter
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runBtnRef.current?.click();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  async function run() {
    setErrorMsg(null);

    // Check authentication first
    if (!session) {
      setErrorMsg("Please sign in to use the playground.");
      return;
    }

    if (!prompt.trim()) {
      setErrorMsg("Prompt cannot be empty.");
      return;
    }

    if (prompt.length > LIMITS.PROMPT_MAX_LENGTH) {
      setErrorMsg(`Prompt is too long (max ${LIMITS.PROMPT_MAX_LENGTH} characters).`);
      return;
    }

    // Cancel any existing request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    setOutput("");
    setIsLoading(true);
    setIsStreaming(false);
    const t0 = performance.now();

    const inTok = roughTokenEstimate(prompt);
    let outputText = ""; // Accumulate output in local variable
    const replay: [number, string][] = [];

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt, temperature, top_p, max_tokens }),
        signal: abortControllerRef.current.signal
      });

      if (!res.ok || !res.body) {
        const msg = await res.text().catch(() => "Unknown error");
        throw new Error(msg || `Bad response ${res.status}`);
      }

      setIsStreaming(true);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += dec.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop() ?? "";

        for (const f of chunks) {
          if (f.startsWith("data: ")) {
            const token = f.slice(6);
            outputText += token; // Accumulate in local variable
            setOutput(outputText); // Update state with complete string
            replay.push([Math.round(performance.now() - t0), token]);
          }
        }
      }

      setIsStreaming(false);

      // Calculate tokens from final output
      const outTok = roughTokenEstimate(outputText);
      const latency = Math.round(performance.now() - t0);
      const cost = estimateCostUSD(model, inTok, outTok);

      // Log the run
      const logRes = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          params: JSON.stringify({ feature: "playground", temperature, top_p, max_tokens }),
          tokensInput: inTok,
          tokensOutput: outTok,
          latencyMs: latency,
          costUsd: cost,
          outputText: outputText,
          replayFrames: JSON.stringify(replay),
          isPublic: false
        })
      });

      if (!logRes.ok) {
        console.error("Failed to log run:", await logRes.text().catch(() => ""));
      }

      setRefreshKey((k) => k + 1);
    } catch (e: any) {
      if (e.name === "AbortError") {
        console.log("Request was cancelled");
        return;
      }
      console.error("Generation error:", e);

      // Parse error message for better UX
      let errorMessage = e.message || "Generation failed. Please try again.";
      try {
        const parsed = JSON.parse(errorMessage);
        errorMessage = parsed.error || errorMessage;
      } catch {
        // Not JSON, use as-is
      }

      setErrorMsg(errorMessage);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }

  // Load previous run configuration
  function loadRun(run: { model: string; prompt: string; params: string }) {
    try {
      const params = JSON.parse(run.params);
      setModel(run.model);
      setPrompt(run.prompt);
      setTemperature(params.temperature ?? 0.7);
      setTopP(params.top_p ?? 1);
      setMaxTokens(params.max_tokens ?? 256);
      setErrorMsg(null);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      console.error("Failed to load run:", e);
    }
  }

  return (
    <main className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-xl font-semibold">Playground</h1>

        {errorMsg && (
          <div
            className="glass-strong border border-red-500/30 bg-red-900/20 text-red-200 px-4 py-3 rounded-xl flex items-start gap-3 shadow-neon-purple animate-pulse"
            role="alert"
          >
            <svg
              className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-red-300/90 flex-1">{errorMsg}</p>
          </div>
        )}

      <Controls
        model={model}
        setModel={setModel}
        temperature={temperature}
        setTemperature={setTemperature}
        top_p={top_p}
        setTopP={setTopP}
        max_tokens={max_tokens}
        setMaxTokens={setMaxTokens}
      />

        <div className="glass-strong rounded-xl p-5">
          <label htmlFor="prompt-input" className="block text-sm mb-2">
            Prompt
          </label>
          <textarea
            id="prompt-input"
            aria-label="Enter your prompt"
            aria-describedby="prompt-hint"
            className="w-full glass border border-slate-600/30 focus-neon p-4 rounded-lg h-40 transition-all font-mono text-sm resize-y text-slate-100 placeholder-slate-500"
            placeholder="Write a haiku about autumn code."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isLoading}
          />
          <p id="prompt-hint" className="sr-only">Press Cmd+Enter (Mac) or Ctrl+Enter (Windows) to submit</p>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <button
            ref={runBtnRef}
            onClick={run}
            disabled={isLoading}
            aria-busy={isLoading}
            className={`px-5 py-2.5 rounded-lg font-semibold transition-all focus-neon border-2 btn-run ${
              isLoading
                ? "glass border-slate-700/60 cursor-not-allowed text-slate-500"
                : "glass border-slate-300/60 hover:border-slate-200 text-slate-100"
            }`}
            title="Run"
          >
            <span>{isStreaming ? "Streaming..." : isLoading ? "Generating..." : "Run"}</span>
            <span className="shine" aria-hidden="true"></span>
          </button>
          {/* Share replay UI intentionally removed */}
        </div>

        <Output text={output} isStreaming={isStreaming} />
        <RunsTable refreshKey={refreshKey} onLoadRun={loadRun} defaultFilter="playground" />
      </div>
    </main>
  );
}
