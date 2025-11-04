# Patch Notes - v0.1.1

## Response to Code Review

This patch addresses all "must-fix" items and implements the suggested improvements from the code review.

## ✅ Must-Fix Items (All Resolved)

### 1. src/types.ts - Type Definition Fixes
**Issue**: Tokenization issues in Swara union type and TranscribeOptions
**Fix**: 
- Ensured all Swara literals are properly quoted
- Completed all type definitions in TranscribeOptions
- Added new options: `snapToleranceCents` and `minNoteEnergy`
- Added documentation for Float32Array default behavior (44100 Hz)

**Files Changed**:
- `src/types.ts`

### 2. src/pipeline/transcribe.ts - Function Signature
**Issue**: AudioBuffer type potentially split across lines
**Fix**:
- Confirmed proper function signature with all input types
- Function correctly accepts: Float32Array | AudioBuffer | ArrayBuffer | Buffer

**Files Changed**:
- `src/pipeline/transcribe.ts`

### 3. src/audio/AudioSource.ts - Buffer Handling
**Issue**: Node Buffer → ArrayBuffer conversion broken
**Fix**:
```typescript
const buffer =
  input instanceof ArrayBuffer
    ? input
    : (input as Buffer).buffer.slice(
        (input as Buffer).byteOffset,
        (input as Buffer).byteOffset + (input as Buffer).byteLength
      );
```

**Files Changed**:
- `src/audio/AudioSource.ts`

## 🛡️ Runtime Correctness Improvements

### Pitch Detector Guards

**YIN Detector** (`src/dsp/pitch/yin.ts`):
- ✅ Added RMS silence detection (threshold: 1e-6) before processing
- ✅ Added bounds checking on detected period
- ✅ Added divide-by-zero protection in parabolic interpolation
```typescript
const denominator = 2 * (2 * s1 - s0 - s2);
if (Math.abs(denominator) < 1e-10) {
  return tau;
}
```

**ACF Detector** (`src/dsp/pitch/acf.ts`):
- ✅ Added RMS silence detection (threshold: 1e-6)

### Note Segmentation (`src/pipeline/postprocess.ts`)
- ✅ Added minimum note energy check using voicing confidence
- ✅ Added outlier detection: >1200 cent jumps treated as new notes
- ✅ Added energy threshold to `finalizeNote` function

### Swara Mapping (`src/pipeline/postprocess.ts`)
- ✅ Implemented raga fallback: when raga-constrained mapping fails, try global swara set with 0.5x confidence penalty
- ✅ Now accepts `snapToleranceCents` parameter (passed through from options)

### WAV Parser (`src/audio/AudioSource.ts`)
- ✅ Improved chunk scanning to handle JUNK, LIST, bext, iXML, etc.
- ✅ Added explicit error for 24-bit PCM: "24-bit PCM is not supported. Please convert to 16-bit or 32-bit float WAV format."
- ✅ Improved error messages for all unsupported formats

### Resampling (`src/audio/resample.ts`)
- ✅ Added boxcar low-pass filter before downsampling to reduce aliasing
- ✅ Filter length based on decimation ratio

## 🎯 API Enhancements

### New AudioSourceInput Type
```typescript
export interface AudioSourceInput {
  samples: Float32Array;
  sampleRate: number;
}
```

**Usage**:
```typescript
const input: AudioSourceInput = {
  samples: myFloat32Array,
  sampleRate: 22050
};
const result = await transcribe(input);
```

**Benefits**:
- Explicit sample rate specification
- Avoids silent 44100 Hz assumption
- Type-safe alternative to plain Float32Array

### New TranscribeOptions
- `snapToleranceCents?: number` - Control swara snap tolerance (default: 50)
- `minNoteEnergy?: number` - Control minimum note energy (default: 0.01)

### Raga Fallback Behavior
When `ragaHint` is provided but a note doesn't fit:
1. First tries raga-constrained swaras
2. If fails, tries global swara set
3. If succeeds with global, confidence multiplied by 0.5
4. Helps with out-of-raga gamakas and transitions

## 📦 Package Improvements

### package.json
- ✅ Added `"sideEffects": false` for better tree-shaking
- ✅ Updated version to 0.1.1

### Exports
- ✅ Exported `AudioSourceInput` type from index.ts

## 📚 Documentation Updates

### README.md
- ✅ Documented AudioSourceInput usage
- ✅ Added new options to API reference
- ✅ Added v0.1.1 feature example

### New Files
- ✅ `CHANGELOG.md` - Complete version history
- ✅ `scripts/generate-test-wavs.ts` - Test fixture generator

## 🧪 Testing Readiness

### Suggested Additional Tests (for future)
While not implemented in this patch, these would be valuable additions:

1. **WAV Decode Tests**:
   - 16-bit mono WAV → verify sampleRate, duration, length
   - 16-bit stereo WAV → verify mono conversion
   - 32-bit float WAV → verify decoding
   - 24-bit PCM WAV → verify error message

2. **Snap Boundary Tests**:
   - Values at midpoints between swaras
   - Values just inside/outside tolerance

3. **Silence Tests**:
   - Silent buffer → no notes, low tonic confidence
   - Already partially covered in existing tests

4. **Tempo Tests**:
   - Click track → verify BPM detection
   - Currently experimental feature

### Current Test Coverage
All existing tests still pass with new changes:
- ✅ pitch.spec.ts - Pitch detection accuracy
- ✅ tonic.spec.ts - Tonic detection
- ✅ swara-map.spec.ts - Swara mapping
- ✅ pipeline.spec.ts - End-to-end

## 🚀 Performance Notes

### Optimizations
- Boxcar filter is simple and fast (moving average)
- RMS checks are O(n) single-pass
- No additional allocations in hot paths

### Memory
- Boxcar filter creates one temporary array during downsampling
- Minimal GC pressure added

## ✨ Quality Improvements

### Error Messages
**Before**: "Unsupported PCM bit depth: 24"
**After**: "24-bit PCM is not supported. Please convert to 16-bit or 32-bit float WAV format."

**Before**: "Unsupported audio format: 3"
**After**: "Unsupported audio format code: 3. Supported: PCM (1) and IEEE Float (3)."

### Code Comments
- Added clarifying comments about word alignment in WAV parsing
- Documented Float32Array behavior in type definitions
- Added JSDoc for new functions

## 📋 Checklist Summary

✅ All must-fix compile blockers resolved
✅ Runtime correctness improvements implemented
✅ Edge cases handled (silence, outliers, divide-by-zero)
✅ API enhancements added (AudioSourceInput, new options)
✅ Raga fallback logic implemented
✅ Anti-aliasing filter added
✅ Package.json updated (sideEffects, version)
✅ Documentation updated (README, CHANGELOG)
✅ Error messages improved
✅ Type exports completed
✅ All existing tests still pass

## 🎯 Result

The library now:
- ✅ Compiles cleanly with no TypeScript errors
- ✅ Handles edge cases robustly
- ✅ Provides better error messages
- ✅ Offers more control via API options
- ✅ Has improved quality with anti-aliasing and outlier detection
- ✅ Maintains backward compatibility (all existing code still works)
- ✅ Is better documented

## 🔄 Migration Guide

### For Existing Users

**No breaking changes!** All existing code continues to work.

**Optional improvements**:

1. **Explicit sample rates**:
```typescript
// Old way (still works)
await transcribe(myFloat32Array);

// New way (more explicit)
await transcribe({ samples: myFloat32Array, sampleRate: 22050 });
```

2. **Fine-tune tolerance**:
```typescript
await transcribe(audio, {
  snapToleranceCents: 40,  // Stricter snapping
  minNoteEnergy: 0.02,     // Higher energy threshold
});
```

3. **Raga fallback**:
```typescript
// Now automatically falls back to global swaras
// with reduced confidence for out-of-raga notes
await transcribe(audio, { ragaHint: "Mohanam" });
```

---

**Version**: 0.1.1  
**Status**: Production Ready  
**All Review Items**: Addressed ✅
