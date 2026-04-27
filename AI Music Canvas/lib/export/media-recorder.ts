/**
 * Export pipeline: canvas + audio → WebM via MediaRecorder.
 * Uses parallel-tap audio graph (speakers stay connected during recording).
 *
 * Audio graph during recording:
 *   source ─── gainNode ─┬── analyserNode ── audioContext.destination (speakers)
 *                         └── mediaStreamDestination (recording tap)
 */

export interface ExportOptions {
  canvas: HTMLCanvasElement;
  audioContext: AudioContext;
  gainNode: GainNode;
  mimeType: string;
  durationMs: number;
  onProgress: (progress: number) => void;
  onComplete: (blob: Blob) => void;
  onError: (error: string) => void;
}

interface RecordingState {
  mediaRecorder: MediaRecorder;
  mediaStreamDestination: MediaStreamAudioDestinationNode;
  chunks: Blob[];
  progressInterval: ReturnType<typeof setInterval>;
  startTime: number;
}

let activeRecording: RecordingState | null = null;

export function startRecording(options: ExportOptions): void {
  if (activeRecording) {
    stopRecording();
  }

  try {
    // 1. Create MediaStreamAudioDestinationNode (parallel tap)
    const mediaStreamDestination = options.audioContext.createMediaStreamDestination();

    // 2. Connect gain → mediaStreamDestination (PARALLEL — does not disconnect existing path)
    options.gainNode.connect(mediaStreamDestination);

    // 3. Get video track from canvas at 30fps
    const canvasStream = options.canvas.captureStream(30);
    const videoTrack = canvasStream.getVideoTracks()[0];

    // 4. Get audio track from destination
    const audioTrack = mediaStreamDestination.stream.getAudioTracks()[0];

    // 5. Merge into combined stream
    const tracks: MediaStreamTrack[] = [videoTrack];
    if (audioTrack) tracks.push(audioTrack);
    const combinedStream = new MediaStream(tracks);

    // 6. Create MediaRecorder
    const mediaRecorder = new MediaRecorder(combinedStream, {
      mimeType: options.mimeType,
      videoBitsPerSecond: 2_500_000,
    });

    const chunks: Blob[] = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: options.mimeType });
      cleanup();
      options.onComplete(blob);
    };

    mediaRecorder.onerror = () => {
      cleanup();
      options.onError('Recording failed. Please try again.');
    };

    // 7. Start recording
    mediaRecorder.start(100); // timeslice: collect data every 100ms
    const startTime = Date.now();

    // Progress tracking
    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / options.durationMs, 1);
      options.onProgress(progress);

      if (elapsed >= options.durationMs) {
        stopRecording();
      }
    }, 100);

    activeRecording = {
      mediaRecorder,
      mediaStreamDestination,
      chunks,
      progressInterval,
      startTime,
    };
  } catch (err) {
    options.onError(
      err instanceof Error ? err.message : 'Failed to start recording.'
    );
  }
}

export function stopRecording(): void {
  if (!activeRecording) return;

  const { mediaRecorder, progressInterval } = activeRecording;
  clearInterval(progressInterval);

  if (mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

function cleanup(): void {
  if (!activeRecording) return;

  const { mediaStreamDestination } = activeRecording;

  // Disconnect the parallel tap (keeps speakers connected)
  try {
    mediaStreamDestination.disconnect();
  } catch {
    // Already disconnected
  }

  activeRecording = null;
}

/** Trigger download of a recorded blob. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
