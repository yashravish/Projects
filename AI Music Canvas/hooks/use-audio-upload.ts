'use client';

import { useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useStudioStore } from '@/store/studio-store';
import { decodeAudioFile, extractWaveformPeaks } from '@/lib/audio/decode';
import { detectSections } from '@/lib/audio/sections';

export function useAudioUpload() {
  const setFile = useStudioStore((s) => s.setFile);
  const setAudioBuffer = useStudioStore((s) => s.setAudioBuffer);
  const setWaveformPeaks = useStudioStore((s) => s.setWaveformPeaks);
  const setDropzoneState = useStudioStore((s) => s.setDropzoneState);
  const setDecodeError = useStudioStore((s) => s.setDecodeError);
  const setSections = useStudioStore((s) => s.setSections);
  const setPlayback = useStudioStore((s) => s.setPlayback);
  const addToast = useStudioStore((s) => s.addToast);
  const file = useStudioStore((s) => s.file);

  const { isLoading, error } = useQuery({
    queryKey: ['decode-audio', file?.name, file?.size],
    queryFn: async () => {
      if (!file) throw new Error('No file');

      setDropzoneState('decoding');

      const result = await decodeAudioFile(file);

      setDropzoneState('analyzing');
      const peaks = extractWaveformPeaks(result.buffer, 1000);
      const sections = detectSections(result.buffer);

      setAudioBuffer(result.buffer);
      setWaveformPeaks(peaks);
      setSections(sections);
      setPlayback({ duration: result.duration });
      setDropzoneState('success');

      return result;
    },
    enabled: !!file,
    retry: false,
    staleTime: Infinity,
  });

  const handleFile = useCallback(
    (uploadedFile: File) => {
      // Validate
      if (!uploadedFile.type.startsWith('audio/')) {
        addToast('Please upload an audio file (MP3, WAV, OGG, etc.)', 'error');
        return;
      }

      if (uploadedFile.size > 25 * 1024 * 1024) {
        addToast('File too large. Maximum size is 25MB.', 'error');
        return;
      }

      setFile(uploadedFile);
    },
    [setFile, addToast]
  );

  // Handle decode errors
  if (error) {
    const msg = error instanceof Error ? error.message : 'Failed to decode audio';
    if (useStudioStore.getState().dropzoneState !== 'error') {
      setDecodeError(msg);
      addToast(`Decode error: ${msg}`, 'error');
    }
  }

  return {
    handleFile,
    isDecoding: isLoading,
  };
}
