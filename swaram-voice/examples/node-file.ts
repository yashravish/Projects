import { readFileSync, writeFileSync } from "node:fs";
import { transcribe, toMIDI, toJSON } from "../src/index";

async function main() {
  console.log("Loading audio file...");
  
  // Read WAV file
  const audioBuffer = readFileSync("tests/fixtures/sa-re-ga.wav");

  console.log("Transcribing...");
  
  // Transcribe with options
  const transcription = await transcribe(audioBuffer, {
    sampleRate: 22050,
    detector: "yin",
    tonicHz: "auto",
    ragaHint: null, // or "Mohanam", "Sankarabharanam", etc.
    tempoBPM: null,
  });

  console.log("\nTranscription complete!");
  console.log(`Detected tonic (Sa): ${transcription.tonic.hz.toFixed(2)} Hz`);
  console.log(`Confidence: ${(transcription.tonic.confidence * 100).toFixed(1)}%`);
  console.log(`Number of swaras: ${transcription.swaras.length}`);

  // Display swaras
  console.log("\nSwara sequence:");
  for (const swara of transcription.swaras) {
    const duration = ((swara.end - swara.start) * 1000).toFixed(0);
    const glideMarker = swara.glide ? "~" : "";
    console.log(
      `  ${swara.swara}${glideMarker} @ ${swara.start.toFixed(2)}s ` +
      `(${duration}ms, conf: ${(swara.confidence * 100).toFixed(0)}%)`
    );
  }

  // Export to JSON
  console.log("\nExporting to JSON...");
  writeFileSync("out/transcription.json", toJSON(transcription));

  // Export to MIDI
  console.log("Exporting to MIDI...");
  const midiBytes = toMIDI(transcription);
  writeFileSync("out/transcription.mid", midiBytes);

  console.log("\nDone! Check out/ directory for results.");
}

// Run
main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
