/**
 * Example: Live microphone transcription (Browser)
 */

import { transcribe } from "../src/index";
import type { Transcription } from "../src/types";

/**
 * Capture audio from microphone and transcribe
 */
export async function captureLiveMicrophone(durationSeconds: number = 5): Promise<Transcription> {
  // Request microphone access
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 44100,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  console.log("Recording for", durationSeconds, "seconds...");

  // Create audio context
  const audioContext = new AudioContext({ sampleRate: 44100 });
  const source = audioContext.createMediaStreamSource(stream);

  // Create processor to capture audio
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];

  // Record audio
  processor.onaudioprocess = (e) => {
    const inputData = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(inputData));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  // Wait for duration
  await new Promise((resolve) => setTimeout(resolve, durationSeconds * 1000));

  // Stop recording
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  await audioContext.close();

  console.log("Recording complete, transcribing...");

  // Combine chunks
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audioBuffer = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    audioBuffer.set(chunk, offset);
    offset += chunk.length;
  }

  // Transcribe
  const transcription = await transcribe(audioBuffer, {
    sampleRate: 22050,
    detector: "yin",
    tonicHz: "auto",
  });

  console.log("Transcription complete!");
  return transcription;
}

/**
 * Display transcription results in the DOM
 */
export function displayTranscription(transcription: Transcription, containerId: string): void {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error("Container not found:", containerId);
    return;
  }

  // Clear container
  container.innerHTML = "";

  // Display tonic
  const tonicDiv = document.createElement("div");
  tonicDiv.className = "tonic-info";
  tonicDiv.innerHTML = `
    <h3>Detected Tonic (Sa)</h3>
    <p>Frequency: ${transcription.tonic.hz.toFixed(2)} Hz</p>
    <p>Confidence: ${(transcription.tonic.confidence * 100).toFixed(1)}%</p>
  `;
  container.appendChild(tonicDiv);

  // Display swaras
  const swarasDiv = document.createElement("div");
  swarasDiv.className = "swaras-list";
  swarasDiv.innerHTML = "<h3>Swara Sequence</h3>";

  const list = document.createElement("ul");
  for (const swara of transcription.swaras) {
    const item = document.createElement("li");
    const duration = ((swara.end - swara.start) * 1000).toFixed(0);
    const glideMarker = swara.glide ? " ~" : "";
    
    item.textContent = `${swara.swara}${glideMarker} @ ${swara.start.toFixed(2)}s ` +
      `(${duration}ms, ${(swara.confidence * 100).toFixed(0)}% conf)`;
    
    // Color by confidence
    const hue = swara.confidence * 120; // Green (120) for high confidence
    item.style.color = `hsl(${hue}, 70%, 40%)`;
    
    list.appendChild(item);
  }

  swarasDiv.appendChild(list);
  container.appendChild(swarasDiv);
}

/**
 * Example usage in HTML page
 */
export async function runDemo(): Promise<void> {
  // Check browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Your browser doesn't support microphone access!");
    return;
  }

  // Create UI
  const app = document.getElementById("app");
  if (!app) return;

  app.innerHTML = `
    <div class="demo-container">
      <h1>@swaram/voice Live Demo</h1>
      <button id="record-btn">Record 5 seconds</button>
      <div id="status"></div>
      <div id="results"></div>
    </div>
  `;

  // Add event listener
  const button = document.getElementById("record-btn");
  const status = document.getElementById("status");

  button?.addEventListener("click", async () => {
    if (!status || !button) return;
    
    button.disabled = true;
    status.textContent = "Recording...";

    try {
      const transcription = await captureLiveMicrophone(5);
      status.textContent = "Transcription complete!";
      displayTranscription(transcription, "results");
    } catch (err) {
      status.textContent = `Error: ${err}`;
      console.error(err);
    } finally {
      button.disabled = false;
    }
  });
}

// Auto-run if loaded as main module in browser
if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", runDemo);
}
