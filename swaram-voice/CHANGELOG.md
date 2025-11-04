# Changelog

All notable changes to @swaram/voice will be documented in this file.

## [0.1.1] - 2024-10-29

### Fixed

#### Critical Compile Fixes
- Fixed `Swara` type definition to ensure all string literals are properly quoted
- Fixed `TranscribeOptions` interface with complete type definitions
- Fixed `AudioSource.ts` Buffer to ArrayBuffer conversion for Node.js compatibility
- Added proper bounds checking in `parabolic interpolation to prevent divide-by-zero

#### Runtime Correctness
- Added silence detection guards in both YIN and ACF pitch detectors to prevent false voicing on silent frames
- Added RMS energy check (threshold: 1e-6) before pitch detection
- Added bounds checking on detected period in YIN detector
- Added outlier detection in note segmentation (>1200 cent jumps treated as new notes)
- Added minimum note energy check to avoid segmenting breath noise
- Improved WAV parser to properly handle non-audio chunks (JUNK, LIST, bext, iXML)
- Added helpful error messages for unsupported WAV formats (24-bit PCM)

#### API Improvements
- Added `AudioSourceInput` type for explicit sample rate specification with Float32Array
- Added `snapToleranceCents` option to `TranscribeOptions` (default: 50)
- Added `minNoteEnergy` option to `TranscribeOptions` (default: 0.01)
- Exported `AudioSourceInput` type from main index
- Added raga fallback logic: out-of-raga notes use global swara set with 0.5x confidence penalty

#### Performance & Quality
- Added simple boxcar anti-aliasing filter before downsampling to reduce aliasing artifacts
- Documented Float32Array default sample rate assumption (44100 Hz) in type definitions
- Added `sideEffects: false` to package.json for better tree-shaking

#### Documentation
- Clarified Float32Array sample rate behavior in `TranscribeOptions` JSDoc
- Added comments about word alignment in WAV chunk parsing
- Improved error messages for unsupported audio formats

### Added
- WAV test fixture generator script (`scripts/generate-test-wavs.ts`)
- Support for out-of-raga note detection with confidence penalty
- Better error handling throughout the pipeline

## [0.1.0] - 2024-10-29

### Added
- Initial L0 release
- YIN and ACF pitch detection algorithms
- Tonic (Sa) detection with confidence scoring
- 12 Carnatic swara support (S, R1, R2, G2, G3, M1, M2, P, D1, D2, N2, N3)
- 10 pre-defined ragas with arohanam/avarohanam patterns
- Coarse gamaka detection (glide vs sustain)
- Basic tempo/tāla detection
- MIDI and JSON export
- Node.js and browser support
- Zero runtime dependencies
- Comprehensive test suite
- Full TypeScript strict mode
- Examples for Node.js and browser
