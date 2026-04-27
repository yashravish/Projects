export function DiagramSVG() {
  return (
    <svg
      viewBox="0 0 800 480"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full h-auto"
      aria-label="Architecture diagram showing data flow from audio file through decode, analysis, and rendering"
    >
      {/* Background grid */}
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width="800" height="480" fill="url(#grid)" />

      {/* Audio File Input */}
      <rect x="20" y="40" width="160" height="60" rx="8" fill="rgba(232,182,90,0.08)" stroke="rgba(232,182,90,0.3)" strokeWidth="1" />
      <text x="100" y="65" textAnchor="middle" fill="#E8B65A" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">AUDIO FILE</text>
      <text x="100" y="82" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">Drop / Browse / Sample</text>

      {/* Arrow down */}
      <line x1="100" y1="100" x2="100" y2="140" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Decode */}
      <rect x="20" y="140" width="160" height="60" rx="8" fill="rgba(111,168,220,0.08)" stroke="rgba(111,168,220,0.3)" strokeWidth="1" />
      <text x="100" y="165" textAnchor="middle" fill="#6FA8DC" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">DECODE</text>
      <text x="100" y="182" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">Web Audio API</text>

      {/* Arrow right from Decode to AudioBuffer */}
      <line x1="180" y1="170" x2="240" y2="170" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* AudioBuffer */}
      <rect x="240" y="140" width="160" height="60" rx="8" fill="rgba(177,156,217,0.08)" stroke="rgba(177,156,217,0.3)" strokeWidth="1" />
      <text x="320" y="165" textAnchor="middle" fill="#B19CD9" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">AUDIO BUFFER</text>
      <text x="320" y="182" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">Float32Array PCM data</text>

      {/* Arrow down from AudioBuffer to Analyzer */}
      <line x1="320" y1="200" x2="320" y2="250" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Arrow right from AudioBuffer to Section Detection */}
      <line x1="400" y1="170" x2="460" y2="170" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Section Detection */}
      <rect x="460" y="140" width="160" height="60" rx="8" fill="rgba(139,111,71,0.08)" stroke="rgba(139,111,71,0.3)" strokeWidth="1" />
      <text x="540" y="165" textAnchor="middle" fill="#8B6F47" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">SECTIONS</text>
      <text x="540" y="182" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">Energy heuristics</text>

      {/* Analyzer */}
      <rect x="240" y="250" width="160" height="60" rx="8" fill="rgba(255,45,122,0.08)" stroke="rgba(255,45,122,0.3)" strokeWidth="1" />
      <text x="320" y="275" textAnchor="middle" fill="#FF2D7A" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">ANALYSER NODE</text>
      <text x="320" y="292" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">FFT → bass/mid/treble</text>

      {/* Arrow down from Analyzer to Renderer */}
      <line x1="320" y1="310" x2="320" y2="360" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Canvas Renderer */}
      <rect x="220" y="360" width="200" height="70" rx="8" fill="rgba(232,182,90,0.08)" stroke="rgba(232,182,90,0.3)" strokeWidth="1" />
      <text x="320" y="385" textAnchor="middle" fill="#E8B65A" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">CANVAS RENDERER</text>
      <text x="320" y="402" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">60fps RAF loop</text>
      <text x="320" y="418" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">Mode → Particles + Waveform</text>

      {/* Zustand Store */}
      <rect x="600" y="250" width="170" height="80" rx="8" fill="rgba(0,240,255,0.06)" stroke="rgba(0,240,255,0.2)" strokeWidth="1" />
      <text x="685" y="275" textAnchor="middle" fill="#00F0FF" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">ZUSTAND STORE</text>
      <text x="685" y="292" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">mode, controls, playback</text>
      <text x="685" y="306" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">getState() in RAF loop</text>
      <text x="685" y="320" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">persist → localStorage</text>

      {/* Arrow from Store to Renderer */}
      <line x1="600" y1="300" x2="420" y2="390" stroke="rgba(0,240,255,0.15)" strokeWidth="1" strokeDasharray="4 4" markerEnd="url(#arrow)" />

      {/* Speakers */}
      <rect x="40" y="360" width="120" height="50" rx="8" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <text x="100" y="385" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">🔊 SPEAKERS</text>
      <text x="100" y="400" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="9">ctx.destination</text>

      {/* Arrow from Analyzer to Speakers */}
      <line x1="240" y1="280" x2="160" y2="370" stroke="rgba(255,255,255,0.1)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Export */}
      <rect x="500" y="370" width="160" height="50" rx="8" fill="rgba(212,175,55,0.08)" stroke="rgba(212,175,55,0.3)" strokeWidth="1" />
      <text x="580" y="393" textAnchor="middle" fill="#D4AF37" fontSize="10" fontFamily="monospace" letterSpacing="0.08em">EXPORT</text>
      <text x="580" y="408" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">captureStream + MediaRecorder</text>

      {/* Arrow from Renderer to Export */}
      <line x1="420" y1="395" x2="500" y2="395" stroke="rgba(255,255,255,0.15)" strokeWidth="1" markerEnd="url(#arrow)" />

      {/* Arrow marker definition */}
      <defs>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 z" fill="rgba(255,255,255,0.3)" />
        </marker>
      </defs>
    </svg>
  );
}
