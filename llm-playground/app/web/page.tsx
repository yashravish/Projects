"use client";
import { useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import Output from "@/components/Output";
import Controls from "@/components/Controls";
import RunsTable from "@/components/RunsTable";
import { estimateCostUSD, roughTokenEstimate } from "@/lib/pricing";
import { DEFAULT_MODEL } from "@/lib/models";
import { LIMITS, WEB_LIMITS } from "@/lib/constants";
import { buildCitedPrompt, CitationSource } from "@/lib/web";

interface WebSource {
  idx: number;
  url: string;
  title: string;
}

interface WebSearchResponse {
  query: string;
  sources: WebSource[];
  snippets: string[];
  usedChars: number;
}

export default function WebPage() {
  const { data: session } = useSession();
  const [model, setModel] = useState<string>(DEFAULT_MODEL);
  const [query, setQuery] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [top_p, setTopP] = useState(1);
  const [max_tokens, setMaxTokens] = useState(512); // Higher default for summaries

  const [sources, setSources] = useState<WebSource[]>([]);
  const [snippets, setSnippets] = useState<string[]>([]);
  const [summary, setSummary] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

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

  async function fetchAndSummarize() {
    setErrorMsg(null);

    // Check authentication first
    if (!session) {
      setErrorMsg("Please sign in to use the web summarizer.");
      return;
    }

    if (!query.trim()) {
      setErrorMsg("Query cannot be empty.");
      return;
    }

    if (query.length > LIMITS.PROMPT_MAX_LENGTH) {
      setErrorMsg(`Query is too long (max ${LIMITS.PROMPT_MAX_LENGTH} characters).`);
      return;
    }

    // Cancel any existing request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    setSources([]);
    setSnippets([]);
    setSummary("");
    setIsLoading(true);
    setIsStreaming(false);
    const t0 = performance.now();

    try {
      // Step 1: Fetch web sources
      const webRes = await fetch("/api/web", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          limit: WEB_LIMITS.WEB_MAX_RESULTS
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!webRes.ok) {
        const errorData = await webRes.json().catch(() => ({ error: "Unknown error" }));
        throw new Error(errorData.error || `Failed to fetch sources: ${webRes.status}`);
      }

      const webData: WebSearchResponse = await webRes.json();

      if (!webData.sources || webData.sources.length === 0) {
        throw new Error("No sources found for this query.");
      }

      setSources(webData.sources);
      setSnippets(webData.snippets);

      // Step 2: Build cited prompt
      const citationSources: CitationSource[] = webData.sources.map((s) => ({
        idx: s.idx,
        url: s.url,
        title: s.title,
      }));

      const prompt = buildCitedPrompt(
        query,
        webData.snippets,
        citationSources,
        WEB_LIMITS.CITED_PROMPT_MAX_LENGTH
      );
      const inTok = roughTokenEstimate(prompt);

      // Step 3: Stream summary from /api/generate
      const genRes = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt, temperature, top_p, max_tokens }),
        signal: abortControllerRef.current.signal,
      });

      if (!genRes.ok || !genRes.body) {
        const msg = await genRes.text().catch(() => "Unknown error");
        throw new Error(msg || `Bad response ${genRes.status}`);
      }

      setIsStreaming(true);
      const reader = genRes.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let outputText = "";
      const replay: [number, string][] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += dec.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop() ?? "";

        for (const f of chunks) {
          if (f.startsWith("data: ")) {
            const token = f.slice(6);
            outputText += token;
            setSummary(outputText);
            replay.push([Math.round(performance.now() - t0), token]);
          }
        }
      }

      setIsStreaming(false);

      // Step 4: Log the run
      const outTok = roughTokenEstimate(outputText);
      const latency = Math.round(performance.now() - t0);
      const cost = estimateCostUSD(model, inTok, outTok);

      const logRes = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt: query, // Store the original query, not the full prompt
          params: JSON.stringify({
            feature: "web",
            temperature,
            top_p,
            max_tokens,
            numSources: webData.sources.length,
            sourceUrls: webData.sources.map((s) => s.url),
          }),
          tokensInput: inTok,
          tokensOutput: outTok,
          latencyMs: latency,
          costUsd: cost,
          outputText: outputText,
          replayFrames: JSON.stringify(replay),
          isPublic: false,
        }),
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
      console.error("Web summarizer error:", e);

      // Parse error message for better UX
      let errorMessage = e.message || "Failed to fetch and summarize. Please try again.";
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
      setQuery(run.prompt); // Restore the query from logged prompt
      setTemperature(params.temperature ?? 0.7);
      setTopP(params.top_p ?? 1);
      setMaxTokens(params.max_tokens ?? 512);
      setErrorMsg(null);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      console.error("Failed to load run:", e);
    }
  }

  return (
    <main className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-semibold">Web Summarizer</h1>
          <p className="text-sm text-slate-400 mt-1">
            Search the web and get AI-powered summaries with citations
          </p>
        </div>

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
          <label htmlFor="query-input" className="block text-sm mb-2">
            Search Query
          </label>
          <input
            id="query-input"
            type="text"
            aria-label="Enter your search query"
            aria-describedby="query-hint"
            className="w-full glass border border-slate-600/30 focus-neon p-4 rounded-lg transition-all font-mono text-sm text-slate-100 placeholder-slate-500"
            placeholder="What are the latest developments in quantum computing?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <p id="query-hint" className="sr-only">
            Press Cmd+Enter (Mac) or Ctrl+Enter (Windows) to submit
          </p>
        </div>

        <button
          ref={runBtnRef}
          onClick={fetchAndSummarize}
          disabled={isLoading}
          aria-busy={isLoading}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all focus-neon border-2 btn-run ${
            isLoading
              ? "glass border-slate-700/60 cursor-not-allowed text-slate-500"
              : "glass border-slate-300/60 hover:border-slate-200 text-slate-100"
          }`}
          title="Fetch & Summarize"
        >
          <span>
            {isStreaming ? "Streaming..." : isLoading ? "Fetching sources..." : "Fetch & Summarize"}
          </span>
          <span className="shine" aria-hidden="true"></span>
        </button>

        {sources.length > 0 && (
          <div className="glass-strong rounded-xl p-5">
            <h2 className="text-sm mb-3 font-semibold">
              Sources ({sources.length})
            </h2>
            <div className="space-y-2">
              {sources.map((source) => (
                <a
                  key={source.idx}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block glass border border-slate-600/30 hover:border-slate-500/50 p-3 rounded-lg transition-all group"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-xs font-mono text-slate-400 flex-shrink-0 mt-0.5">
                      [{source.idx}]
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-200 group-hover:text-white truncate">
                        {source.title}
                      </p>
                      <p className="text-xs text-slate-400 truncate mt-0.5">
                        {source.url}
                      </p>
                    </div>
                    <svg
                      className="w-4 h-4 text-slate-400 group-hover:text-slate-300 flex-shrink-0 mt-1"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </div>
                </a>
              ))}
            </div>
          </div>
        )}

        <Output text={summary} isStreaming={isStreaming} />
        <RunsTable refreshKey={refreshKey} onLoadRun={loadRun} defaultFilter="web" />
      </div>
    </main>
  );
}
