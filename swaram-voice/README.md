# @swaram/voice

A TypeScript/JavaScript library for transcribing Carnatic vocal music into swara notation.

## Features

- **Pitch Detection**: YIN and ACF algorithms for robust f0 extraction
- **Tonic Detection**: Automatic Sa detection with confidence scoring
- **Swara Mapping**: Converts frequencies to Carnatic swaras (S, R1, R2, G2, G3, M1, M2, P, D1, D2, N2, N3)
- **Raga Support**: Optional raga hints for improved accuracy
- **Gamaka Detection**: Coarse detection of glides vs sustained notes
- **Tempo Estimation**: Basic tāla/beat detection
- **Export Formats**: JSON and MIDI
- **Cross-Platform**: Works in Node.js and modern browsers

## Installation

```bash
npm install @swaram/voice
```

## Quick Start

### Node.js Example

```typescript
import { readFileSync, writeFileSync } from "node:fs";
import { transcribe, toMIDI, toJSON } from "@swaram/voice";

// Load audio file
const audioBuffer = readFileSync("recording.wav");

// Transcribe
const transcription = await transcribe(audioBuffer, {
  sampleRate: 22050,
  detector: "yin",
  tonicHz: "auto", // or specific Hz value
  ragaHint: "Mohanam", // optional
});

// Display results
console.log(`Tonic: ${transcription.tonic.hz.toFixed(2)} Hz`);
console.log(`Swaras:`, transcription.swaras.map(s => s.swara).join(" "));

// Export
writeFileSync("output.json", toJSON(transcription));
writeFileSync("output.mid", toMIDI(transcription));
```

**New in v0.1.1:** For explicit sample rate control with Float32Array:
```typescript
import type { AudioSourceInput } from "@swaram/voice";

const audioInput: AudioSourceInput = {
  samples: myFloat32Array,
  sampleRate: 44100  // Explicit rate instead of default 44100
};

const result = await transcribe(audioInput);
```

### Browser Example

```typescript
import { transcribe } from "@swaram/voice";

// Get microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audioContext = new AudioContext();
const source = audioContext.createMediaStreamSource(stream);

// Record audio (simplified - see examples/web-mic.ts for full implementation)
// ... capture audio into Float32Array ...

// Transcribe
const transcription = await transcribe(audioBuffer, {
  detector: "yin",
  tonicHz: "auto",
});

console.log("Detected swaras:", transcription.swaras);
```

## API Reference

### Main Functions

#### `transcribe(audio, options?)`

Transcribe audio to swara notation.

**Parameters:**
- `audio`: `Float32Array | AudioBuffer | ArrayBuffer | Buffer | AudioSourceInput` - Audio input
  - `Float32Array`: Mono PCM samples (assumes 44100 Hz, use `AudioSourceInput` for explicit rate)
  - `AudioSourceInput`: `{ samples: Float32Array, sampleRate: number }` for explicit sample rate
  - `AudioBuffer`: Web Audio API buffer
  - `ArrayBuffer` or `Buffer`: WAV file bytes
- `options`: `TranscribeOptions` - Configuration options

**Options:**
```typescript
{
  sampleRate?: number;        // Target sample rate (default: 22050)
  frameSize?: number;         // Analysis frame size (default: 2048)
  hopSize?: number;           // Hop size (default: 512)
  detector?: "yin" | "acf";   // Pitch detector (default: "yin")
  ragaHint?: string | null;   // Raga name for constraints
  tempoBPM?: number | "auto"; // Tempo (default: null)
  tonicHz?: Hz | "auto";      // Tonic frequency (default: "auto")
  minVoicing?: number;        // Min voicing confidence (default: 0.3)
  snapToleranceCents?: number; // Swara snap tolerance (default: 50)
  minNoteEnergy?: number;     // Min note energy threshold (default: 0.01)
}
```

**Returns:** `Promise<Transcription>`

```typescript
{
  tonic: {
    hz: number;           // Detected tonic frequency
    confidence: number;   // 0-1 confidence score
  };
  swaras: Array<{
    start: number;        // Start time (seconds)
    end: number;          // End time (seconds)
    swara: Swara;         // Swara name
    centsFromSa: number;  // Cents offset from Sa
    confidence: number;   // 0-1 confidence score
    glide?: boolean;      // True if gamaka detected
  }>;
  notes: Array<...>;      // Intermediate note events
  tempo?: number;         // Detected tempo (BPM)
}
```

#### `toMIDI(transcription, baseMIDINote?)`

Export transcription to Standard MIDI File.

**Parameters:**
- `transcription`: `Transcription` - Transcription result
- `baseMIDINote`: `number` - MIDI note for Sa (default: 60 = C4)

**Returns:** `Uint8Array` - MIDI file bytes

#### `toJSON(transcription, pretty?)`

Export transcription to JSON string.

**Parameters:**
- `transcription`: `Transcription` - Transcription result
- `pretty`: `boolean` - Pretty-print (default: true)

**Returns:** `string`

### Types

```typescript
type Swara = "S" | "R1" | "R2" | "G2" | "G3" | "M1" | "M2" | "P" | "D1" | "D2" | "N2" | "N3";
```

## Supported Ragas

The library includes definitions for common Carnatic ragas:

- Mohanam
- Sankarabharanam
- Kalyani
- Mayamalavagowla
- Kharaharapriya
- Hamsadhwani
- Bhairavi
- Shanmukhapriya
- Thodi
- Abhogi

Use raga hints to improve transcription accuracy:

```typescript
await transcribe(audio, { ragaHint: "Mohanam" });
```

## Performance

- **Latency**: Typical 10-second clip transcribes in < 1.5 seconds on modern hardware
- **Accuracy**: ≥85% swara accuracy on clean recordings with minimal gamakas
- **Tonic Detection**: Within ±10 cents on test cases

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Watch mode
npm run test:watch

# Lint
npm run lint

# Format
npm run format
```

## Project Structure

```
src/
├── audio/          # Audio I/O and preprocessing
├── dsp/            # Signal processing (pitch, onset)
├── theory/         # Music theory (swara, raga, tala)
├── pipeline/       # Main transcription pipeline
├── export/         # Output formats (MIDI, JSON)
├── types.ts        # TypeScript definitions
└── index.ts        # Public API

examples/
├── node-file.ts    # File processing example
└── web-mic.ts      # Live microphone example

tests/
├── pitch.spec.ts   # Pitch detection tests
├── tonic.spec.ts   # Tonic detection tests
├── swara-map.spec.ts   # Swara mapping tests
└── pipeline.spec.ts    # End-to-end tests
```

## Limitations (L0)

- **Gamakas**: Only coarse detection (glide vs sustain)
- **Polyphony**: Mono audio only (single melody line)
- **Tāla**: Basic tempo detection, no complex rhythm analysis
- **Accuracy**: Best with clean, solo vocal recordings

## Future Roadmap

- [ ] Fine-grained gamaka classification
- [ ] Raga-aware Viterbi decoding
- [ ] Enhanced tāla detection with user taps
- [ ] Support for more ragas and melakartas
- [ ] Real-time streaming mode
- [ ] MusicXML export

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Acknowledgments

- YIN algorithm: Cheveigné & Kawahara (2002)
- Carnatic music theory: Various traditional sources

---

Built with ❤️ for Carnatic music
