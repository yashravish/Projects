const sections = [
  {
    id: 'app-router',
    title: 'App Router & Server Components',
    content: `AI Music Canvas uses Next.js App Router with a deliberate RSC-first strategy. The landing page and architecture page render as server components — zero client JavaScript for static content. The studio page uses \`next/dynamic\` with \`ssr: false\` to load the heavy canvas/audio engine only when needed, keeping the landing page bundle under 120KB.

Each route has its own \`error.tsx\` boundary so a canvas crash doesn't take down the page, and \`loading.tsx\` provides a designed skeleton during streaming SSR.`,
    code: `// app/studio/page.tsx
const StudioClient = dynamic(
  () => import('./studio-client'),
  { ssr: false, loading: () => <Loader /> }
);`,
  },
  {
    id: 'state',
    title: 'State Architecture',
    content: `All client state lives in a single Zustand store, sliced by concern: file, playback, mode, controls, export, and compat. The store uses \`persist\` middleware with \`skipHydration: true\` to avoid SSR hydration mismatches — rehydration happens in a client-only \`useEffect\`.

The critical invariant: the AudioBuffer (~40MB) lives in the store but is never subscribed to via React selectors. The canvas renderer reads it via \`store.getState()\` inside the RAF loop, preventing React re-renders during animation.`,
    code: `// Renderer reads store without React subscription
function renderFrame() {
  const { mode, controls } = useStudioStore.getState();
  const data = analyzerRef.current?.getFrequencyData();
  renderMode(ctx, mode, w, h, data, controls, time);
  requestAnimationFrame(renderFrame);
}`,
  },
  {
    id: 'audio',
    title: 'Audio Pipeline',
    content: `The Web Audio API powers the audio analysis chain. An AudioContext is created lazily on the first user gesture (for Safari compatibility), with a \`webkitAudioContext\` fallback for older browsers.

The audio graph routes through: BufferSourceNode → GainNode → AnalyserNode → destination (speakers). During recording, a parallel MediaStreamAudioDestinationNode taps the GainNode output without disconnecting the speakers.

Section detection uses energy heuristics against a typical song-structure template — not ML, but good enough to demonstrate the feature.`,
    code: `// Audio graph during recording (parallel tap)
// source → gain ─┬─ analyser → speakers
//                 └─ mediaStreamDest (record)
gainNode.connect(analyserNode);
gainNode.connect(mediaStreamDestNode);`,
  },
  {
    id: 'render',
    title: 'Render Loop',
    content: `The canvas renderer owns a single \`requestAnimationFrame\` loop — not React. The \`use-canvas-visualizer\` hook is a thin bridge that calls \`renderer.start()\` in a \`useEffect\` and \`renderer.stop()\` on cleanup.

Each frame: read \`getState()\` for controls, read analyzer for frequency data, delegate to the active mode's render function. Mode transitions crossfade over ~600ms using an offscreen canvas.

DPR-aware sizing ensures retina sharpness: \`canvas.width = clientWidth × devicePixelRatio\`, with \`ctx.scale(dpr, dpr)\`. A ResizeObserver (debounced via RAF) handles layout changes.`,
    code: `// DPR-aware canvas sizing
const dpr = window.devicePixelRatio || 1;
canvas.width = w * dpr;
canvas.height = h * dpr;
ctx.scale(dpr, dpr);`,
  },
  {
    id: 'perf',
    title: 'Performance',
    content: `Key performance decisions:

• Canvas renders at 60fps; export captures at 30fps (halves file size).
• React never re-renders during the animation loop — all hot-path reads go through \`store.getState()\` and refs.
• The VisualCanvas component is wrapped in \`React.memo\` and receives only primitives.
• Sliders update the Zustand store, which the renderer reads on the next frame — no re-render.
• Particle system uses object pooling (recycle dead particles instead of allocating new ones).
• Lazy-loaded studio chunk keeps the landing page under 120KB initial JS.`,
    code: null,
  },
  {
    id: 'a11y',
    title: 'Accessibility',
    content: `• All interactive elements have custom focus rings (\`:focus-visible\`, not \`:focus\`).
• Sliders use full ARIA: \`aria-valuemin\`, \`aria-valuemax\`, \`aria-valuenow\`, \`aria-valuetext\`.
• The canvas has a dynamic \`aria-label\` that updates with the current mode and section.
• Drag-and-drop has a visible "Browse files" button as the keyboard-accessible path.
• \`prefers-reduced-motion\` disables all particles and shows waveform-only rendering.
• Cursor hides after 2s of inactivity during playback (not blanket \`cursor: none\`), disabled on touch devices.
• Keyboard shortcuts: Space (play/pause), arrows (seek), 1-4 (modes), ? (help modal).`,
    code: null,
  },
];

export function ArchitectureExplainer() {
  return (
    <div className="space-y-12">
      {sections.map((section) => (
        <section key={section.id} className="space-y-4">
          <h2 className="font-display text-2xl sm:text-3xl" style={{ color: 'var(--foreground)' }}>
            {section.title}
          </h2>

          <div
            className="text-sm leading-relaxed whitespace-pre-line"
            style={{ color: 'rgba(255,255,255,0.55)' }}
          >
            {section.content}
          </div>

          {section.code && (
            <div
              className="rounded-[var(--radius-card)] overflow-hidden"
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div
                className="px-4 py-2 text-[10px] uppercase tracking-wider"
                style={{
                  color: 'rgba(255,255,255,0.3)',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                }}
              >
                Code
              </div>
              <pre className="p-4 overflow-x-auto">
                <code
                  className="font-mono text-xs leading-relaxed"
                  style={{ color: 'var(--accent)' }}
                >
                  {section.code}
                </code>
              </pre>
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
