# Getting Started with @swaram/voice

## Installation & Setup

1. **Extract the archive:**
   ```bash
   tar -xzf swaram-voice.tar.gz
   cd swaram-voice
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Build the library:**
   ```bash
   npm run build
   ```

4. **Run tests:**
   ```bash
   npm test
   ```

## File Structure Summary

### Core Library (`src/`)
- **types.ts** - All TypeScript type definitions
- **index.ts** - Main entry point and public API

#### Audio Processing (`src/audio/`)
- **AudioSource.ts** - Unified audio input handling (WAV decoder, AudioBuffer)
- **resample.ts** - Audio resampling with linear interpolation
- **windowing.ts** - Window functions (Hann, Hamming, Blackman)

#### DSP (`src/dsp/`)
- **pitch/yin.ts** - YIN pitch detector (robust, recommended)
- **pitch/acf.ts** - ACF pitch detector (faster, simpler)
- **pitch/types.ts** - Pitch detector interfaces
- **onset.ts** - Onset detection for note boundaries
- **smoothing.ts** - Pitch track smoothing and filtering

#### Music Theory (`src/theory/`)
- **tonic.ts** - Tonic (Sa) detection with histogram method
- **swara-map.ts** - Cents-to-swara mapping with Carnatic intervals
- **ragas.ts** - Raga definitions (10 common ragas)
- **tala.ts** - Basic tempo/rhythm detection

#### Pipeline (`src/pipeline/`)
- **transcribe.ts** - Main end-to-end transcription pipeline
- **postprocess.ts** - Note segmentation and swara conversion

#### Export (`src/export/`)
- **midi.ts** - Standard MIDI File export
- **json.ts** - JSON import/export

### Examples (`examples/`)
- **node-file.ts** - Process WAV files from disk
- **web-mic.ts** - Live microphone capture in browser

### Tests (`tests/`)
- **pitch.spec.ts** - Pitch detector tests (YIN, ACF)
- **tonic.spec.ts** - Tonic detection tests
- **swara-map.spec.ts** - Swara mapping tests
- **pipeline.spec.ts** - End-to-end integration tests

## Quick Usage Examples

### 1. Transcribe a WAV file

```typescript
import { readFileSync } from "fs";
import { transcribe } from "@swaram/voice";

const audio = readFileSync("recording.wav");
const result = await transcribe(audio, {
  detector: "yin",
  tonicHz: "auto",
});

console.log("Detected swaras:", result.swaras.map(s => s.swara));
```

### 2. With raga hint

```typescript
const result = await transcribe(audio, {
  ragaHint: "Mohanam",  // Restricts to S R2 G3 P D2
  tonicHz: 220,          // Or "auto" to detect
});
```

### 3. Export to MIDI

```typescript
import { toMIDI } from "@swaram/voice";
import { writeFileSync } from "fs";

const midiBytes = toMIDI(result);
writeFileSync("output.mid", midiBytes);
```

### 4. Browser microphone

```typescript
// Capture 5 seconds
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
// ... capture to Float32Array (see examples/web-mic.ts)

const result = await transcribe(audioBuffer);
```

## Key Algorithms

### Pitch Detection (YIN)
1. Difference function calculation
2. Cumulative mean normalized difference (CMND)
3. Absolute threshold search
4. Parabolic interpolation for sub-sample precision

### Tonic Detection
1. Build pitch histogram in log-frequency space
2. Gaussian smoothing
3. Peak detection
4. Mean-shift refinement
5. Validation using pitch-class profile

### Swara Mapping
1. Convert f0 to cents from Sa: `cents = 1200 * log2(f0/tonic)`
2. Snap to nearest swara within tolerance (±50 cents)
3. Apply raga constraints if provided
4. Calculate confidence based on distance

## Configuration Options

```typescript
interface TranscribeOptions {
  sampleRate?: number;        // 22050 (default), 16000, or 44100
  frameSize?: number;         // 2048 (default), 1024, or 4096
  hopSize?: number;           // 512 (default) - smaller = more frames
  detector?: "yin" | "acf";   // "yin" (default) - more accurate
  ragaHint?: string | null;   // "Mohanam", "Sankarabharanam", etc.
  tempoBPM?: number | "auto"; // null (default), number, or "auto"
  tonicHz?: Hz | "auto";      // "auto" (default) or specific Hz
  minVoicing?: number;        // 0.3 (default) - threshold for voiced
}
```

## Performance Tips

1. **Use YIN for accuracy**, ACF for speed
2. **Lower sample rate** (16000 Hz) for faster processing
3. **Larger hop size** (1024) reduces frame count
4. **Provide tonic** if known instead of "auto"
5. **Use raga hints** for better accuracy

## Testing

```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
npm run test:coverage # Coverage report
```

## Common Issues

### "No voiced frames detected"
- Check audio volume/quality
- Reduce `minVoicing` threshold
- Ensure mono audio

### "Incorrect tonic detected"
- Provide `tonicHz` manually
- Ensure recording has strong Sa references
- Check for background noise

### "Wrong swaras detected"
- Try different `detector` ("yin" vs "acf")
- Adjust `frameSize` and `hopSize`
- Use `ragaHint` if known
- Check if gamakas are confusing the detector

## Next Steps

1. Try the examples in `examples/`
2. Run the test suite to verify installation
3. Experiment with your own audio files
4. Adjust options for your use case
5. Contribute improvements!

## Support

- GitHub: [issues](https://github.com/yourusername/swaram-voice/issues)
- Docs: See README.md for full API reference

---

Happy transcribing! 🎵
