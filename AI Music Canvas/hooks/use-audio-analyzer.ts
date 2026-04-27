'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useStudioStore } from '@/store/studio-store';
import { getAudioContext, resumeAudioContext } from '@/lib/audio/audio-context';
import { createAnalyzer, type AnalyzerSetup } from '@/lib/audio/analyzer';
import type { FrequencyData } from '@/types/audio';

export function useAudioAnalyzer() {
  const analyzerRef = useRef<AnalyzerSetup | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const startTimeRef = useRef(0);
  const pauseTimeRef = useRef(0);

  const playback = useStudioStore((s) => s.playback);
  const setPlayback = useStudioStore((s) => s.setPlayback);
  const setCurrentTime = useStudioStore((s) => s.setCurrentTime);

  const getFrequencyData = useCallback((): FrequencyData | null => {
    if (!analyzerRef.current) return null;
    return analyzerRef.current.getFrequencyData();
  }, []);

  const play = useCallback(async (buffer: AudioBuffer, startFrom: number = 0) => {
    const ctx = await resumeAudioContext();

    // Stop any previous source
    if (sourceRef.current) {
      try { sourceRef.current.stop(); } catch { /* already stopped */ }
    }

    // Setup audio graph
    if (!analyzerRef.current) {
      analyzerRef.current = createAnalyzer(ctx);
    }

    if (!gainRef.current) {
      gainRef.current = ctx.createGain();
      gainRef.current.connect(analyzerRef.current.analyser);
      analyzerRef.current.analyser.connect(ctx.destination);
    }

    gainRef.current.gain.value = playback.isMuted ? 0 : playback.volume;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(gainRef.current);

    const offset = Math.max(0, Math.min(startFrom, buffer.duration));
    source.start(0, offset);
    sourceRef.current = source;
    startTimeRef.current = ctx.currentTime - offset;

    source.onended = () => {
      const elapsed = ctx.currentTime - startTimeRef.current;
      if (elapsed >= buffer.duration - 0.1) {
        setPlayback({ isPlaying: false, currentTime: 0 });
        pauseTimeRef.current = 0;
      }
    };

    setPlayback({ isPlaying: true, duration: buffer.duration });
  }, [playback.isMuted, playback.volume, setPlayback]);

  const pause = useCallback(() => {
    if (sourceRef.current) {
      const ctx = getAudioContext();
      pauseTimeRef.current = ctx.currentTime - startTimeRef.current;
      try { sourceRef.current.stop(); } catch { /* already stopped */ }
      sourceRef.current = null;
    }
    setPlayback({ isPlaying: false });
  }, [setPlayback]);

  const seek = useCallback((time: number) => {
    const buffer = useStudioStore.getState().audioBuffer;
    if (!buffer) return;
    pauseTimeRef.current = time;
    setCurrentTime(time);
    if (playback.isPlaying) {
      play(buffer, time);
    }
  }, [playback.isPlaying, play, setCurrentTime]);

  const setVolume = useCallback((volume: number) => {
    if (gainRef.current) {
      gainRef.current.gain.value = volume;
    }
    setPlayback({ volume });
  }, [setPlayback]);

  const toggleMute = useCallback(() => {
    const newMuted = !playback.isMuted;
    if (gainRef.current) {
      gainRef.current.gain.value = newMuted ? 0 : playback.volume;
    }
    setPlayback({ isMuted: newMuted });
  }, [playback.isMuted, playback.volume, setPlayback]);

  // Update currentTime continuously during playback
  useEffect(() => {
    if (!playback.isPlaying) return;

    let raf: number;
    function tick() {
      const ctx = getAudioContext();
      const current = ctx.currentTime - startTimeRef.current;
      setCurrentTime(current);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(raf);
  }, [playback.isPlaying, setCurrentTime]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (sourceRef.current) {
        try { sourceRef.current.stop(); } catch { /* */ }
      }
      if (analyzerRef.current) {
        analyzerRef.current.disconnect();
      }
    };
  }, []);

  return {
    getFrequencyData,
    analyzerRef,
    gainRef,
    play,
    pause,
    seek,
    setVolume,
    toggleMute,
    pauseTimeRef,
  };
}
