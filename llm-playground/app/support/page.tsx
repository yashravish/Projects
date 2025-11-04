"use client";
import { useEffect, useRef, useState } from "react";
import Output from "@/components/Output";
import Controls from "@/components/Controls";
import RunsTable from "@/components/RunsTable";
import { estimateCostUSD, roughTokenEstimate } from "@/lib/pricing";
import { DEFAULT_MODEL } from "@/lib/models";
import { LIMITS, SUPPORT_LIMITS } from "@/lib/constants";
import { retrieve } from "@/lib/retrieval";

export default function SupportPage() {
  const [model, setModel] = useState<string>(DEFAULT_MODEL);
  const [kbText, setKbText] = useState("");
  const [question, setQuestion] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [top_p, setTopP] = useState(1);
  const [max_tokens, setMaxTokens] = useState(512);

  const [contextChunks, setContextChunks] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [showContext, setShowContext] = useState(false);

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

  async function answerQuestion() {
    setErrorMsg(null);

    // Validation
    if (!kbText.trim()) {
      setErrorMsg("Knowledge base cannot be empty. Please paste your documentation.");
      return;
    }

    if (!question.trim()) {
      setErrorMsg("Question cannot be empty.");
      return;
    }

    if (kbText.length > SUPPORT_LIMITS.KB_TEXT_MAX_LENGTH) {
      setErrorMsg(
        `Knowledge base is too long (max ${SUPPORT_LIMITS.KB_TEXT_MAX_LENGTH.toLocaleString()} characters). Current: ${kbText.length.toLocaleString()}`
      );
      return;
    }

    if (question.length > LIMITS.PROMPT_MAX_LENGTH) {
      setErrorMsg(`Question is too long (max ${LIMITS.PROMPT_MAX_LENGTH} characters).`);
      return;
    }

    // Cancel any existing request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    setContextChunks([]);
    setAnswer("");
    setIsLoading(true);
    setIsStreaming(false);
    const t0 = performance.now();

    try {
      // Step 1: Retrieve relevant chunks
      const { chunks } = retrieve(question, kbText, SUPPORT_LIMITS.TOP_K_CHUNKS);

      if (chunks.length === 0) {
        throw new Error(
          "No relevant context found in the knowledge base. Try rephrasing your question or adding more content."
        );
      }

      setContextChunks(chunks);

      // Step 2: Build strict prompt
      const contextSection = chunks
        .map((chunk, idx) => `[Chunk ${idx + 1}]\n${chunk}`)
        .join('\n\n---\n\n');

      const systemInstructions = `You are a helpful assistant that answers questions ONLY using the provided context chunks. Follow these rules strictly:

1. Only use information from the context chunks below
2. If the answer is not in the context, say "I don't know based on the provided documentation"
3. Cite chunk numbers like [Chunk 1] when referencing information
4. Do not make up information or use knowledge outside the provided context
5. Be concise and direct in your answers

## Context from Knowledge Base

${contextSection}

## User Question

${question}

## Your Answer

Please provide a concise answer based ONLY on the context above:`;

      // Check prompt length
      if (systemInstructions.length > SUPPORT_LIMITS.SUPPORT_PROMPT_MAX_LENGTH) {
        // Try with fewer chunks
        const reducedChunks = chunks.slice(0, 3);
        const reducedContext = reducedChunks
          .map((chunk, idx) => `[Chunk ${idx + 1}]\n${chunk}`)
          .join('\n\n---\n\n');

        const reducedPrompt = `You are a helpful assistant that answers questions ONLY using the provided context chunks. Follow these rules strictly:

1. Only use information from the context chunks below
2. If the answer is not in the context, say "I don't know based on the provided documentation"
3. Cite chunk numbers like [Chunk 1] when referencing information
4. Do not make up information or use knowledge outside the provided context
5. Be concise and direct in your answers

## Context from Knowledge Base

${reducedContext}

## User Question

${question}

## Your Answer

Please provide a concise answer based ONLY on the context above:`;

        if (reducedPrompt.length > LIMITS.PROMPT_MAX_LENGTH) {
          throw new Error(
            "Context is too large even with reduced chunks. Try a more specific question or shorter documentation."
          );
        }

        setContextChunks(reducedChunks); // Update to show only what's being used
      }

      const finalPrompt =
        systemInstructions.length <= SUPPORT_LIMITS.SUPPORT_PROMPT_MAX_LENGTH
          ? systemInstructions
          : systemInstructions.substring(0, SUPPORT_LIMITS.SUPPORT_PROMPT_MAX_LENGTH);

      const inTok = roughTokenEstimate(finalPrompt);

      // Step 3: Stream answer from /api/generate
      const genRes = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt: finalPrompt, temperature, top_p, max_tokens }),
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
            setAnswer(outputText);
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
          prompt: question, // Store the question, not the full prompt
          params: JSON.stringify({
            feature: "support",
            temperature,
            top_p,
            max_tokens,
            contextChunkCount: contextChunks.length,
            chunkPreviews: contextChunks.map((c) => c.substring(0, 120) + "..."),
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
      console.error("Support chat error:", e);

      // Parse error message for better UX
      let errorMessage = e.message || "Failed to answer question. Please try again.";
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
      setQuestion(run.prompt); // Restore the question from logged prompt
      setTemperature(params.temperature ?? 0.7);
      setTopP(params.top_p ?? 1);
      setMaxTokens(params.max_tokens ?? 512);
      setErrorMsg(null);
      // Note: We can't restore kbText as it's not logged (privacy)
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      console.error("Failed to load run:", e);
    }
  }

  return (
    <main className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-semibold">Support Chat</h1>
          <p className="text-sm text-slate-400 mt-1">
            Paste your documentation and ask questions. AI answers using only your content.
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
            <div className="flex-1">
              <p className="font-semibold text-sm mb-1">Error Detected</p>
              <p className="text-sm text-red-300/90">{errorMsg}</p>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-red-400 hover:text-red-300 transition-all hover:scale-110"
              aria-label="Dismiss error"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
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
          <label htmlFor="kb-input" className="block text-sm mb-2">
            Knowledge Base
            <span className="text-slate-400 ml-2 font-normal">
              ({kbText.length.toLocaleString()} / {SUPPORT_LIMITS.KB_TEXT_MAX_LENGTH.toLocaleString()})
            </span>
          </label>
          <textarea
            id="kb-input"
            aria-label="Paste your documentation here"
            className="w-full glass border border-slate-600/30 focus-neon p-4 rounded-lg h-64 transition-all font-mono text-sm resize-y text-slate-100 placeholder-slate-500"
            placeholder="Paste your internal documentation, API docs, or support articles here..."
            value={kbText}
            onChange={(e) => setKbText(e.target.value)}
            disabled={isLoading}
          />
        </div>

        <div className="glass-strong rounded-xl p-5">
          <label htmlFor="question-input" className="block text-sm mb-2">
            Your Question
          </label>
          <input
            id="question-input"
            type="text"
            aria-label="Ask a question about your documentation"
            aria-describedby="question-hint"
            className="w-full glass border border-slate-600/30 focus-neon p-4 rounded-lg transition-all font-mono text-sm text-slate-100 placeholder-slate-500"
            placeholder="How do I configure authentication?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={isLoading}
          />
          <p id="question-hint" className="sr-only">
            Press Cmd+Enter (Mac) or Ctrl+Enter (Windows) to submit
          </p>
        </div>

        <button
          ref={runBtnRef}
          onClick={answerQuestion}
          disabled={isLoading}
          aria-busy={isLoading}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all focus-neon border-2 btn-run ${
            isLoading
              ? "glass border-slate-700/60 cursor-not-allowed text-slate-500"
              : "glass border-slate-300/60 hover:border-slate-200 text-slate-100"
          }`}
          title="Answer"
        >
          <span>{isStreaming ? "Streaming..." : isLoading ? "Finding answer..." : "Answer"}</span>
          <span className="shine" aria-hidden="true"></span>
        </button>

        {contextChunks.length > 0 && (
          <div className="glass-strong rounded-xl p-5">
            <button
              onClick={() => setShowContext(!showContext)}
              className="flex items-center justify-between w-full text-left"
              aria-expanded={showContext}
            >
              <h2 className="text-sm font-semibold">
                Context Used ({contextChunks.length} chunks)
              </h2>
              <svg
                className={`w-5 h-5 text-slate-400 transition-transform ${
                  showContext ? "rotate-180" : ""
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>

            {showContext && (
              <div className="mt-4 space-y-3">
                {contextChunks.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="glass border border-slate-600/30 p-3 rounded-lg"
                  >
                    <div className="text-xs font-mono text-slate-400 mb-2">
                      Chunk {idx + 1}
                    </div>
                    <pre className="text-sm text-slate-200 whitespace-pre-wrap font-mono">
                      {chunk}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <Output text={answer} isStreaming={isStreaming} />
        <RunsTable refreshKey={refreshKey} onLoadRun={loadRun} defaultFilter="support" />
      </div>
    </main>
  );
}
