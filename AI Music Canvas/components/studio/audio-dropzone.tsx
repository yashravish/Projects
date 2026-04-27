'use client';

import { useCallback, useState, useRef } from 'react';
import { Upload, Music, FileAudio } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useStudioStore } from '@/store/studio-store';
import { useAudioUpload } from '@/hooks/use-audio-upload';
import { Loader } from '@/components/ui/loader';
import { formatFileSize } from '@/lib/utils/format';
import type { DropzoneState } from '@/types/studio';

export function AudioDropzone() {
  const dropzoneState = useStudioStore((s) => s.dropzoneState);
  const fileName = useStudioStore((s) => s.fileName);
  const fileSize = useStudioStore((s) => s.fileSize);
  const decodeError = useStudioStore((s) => s.decodeError);
  const clearFile = useStudioStore((s) => s.clearFile);
  const { handleFile } = useAudioUpload();

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFile(files[0]);
      }
    },
    [handleFile]
  );

  const onBrowse = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
    },
    [handleFile]
  );

  const handleSample = useCallback(async () => {
    try {
      const response = await fetch('/demo.mp3');
      const blob = await response.blob();
      const file = new File([blob], 'demo-sample.mp3', { type: 'audio/mpeg' });
      handleFile(file);
    } catch {
      // Sample not available — that's ok
    }
  }, [handleFile]);

  const stateMap: Record<DropzoneState, React.ReactNode> = {
    idle: (
      <div className="flex flex-col items-center gap-4">
        <div
          className="w-14 h-14 rounded-xl flex items-center justify-center"
          style={{
            background: isDragOver ? 'rgba(var(--accent-rgb), 0.15)' : 'rgba(255,255,255,0.04)',
            border: `1px solid ${isDragOver ? 'rgba(var(--accent-rgb), 0.3)' : 'rgba(255,255,255,0.06)'}`,
            transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          <Upload size={22} strokeWidth={1.5} style={{ color: isDragOver ? 'var(--accent)' : 'rgba(255,255,255,0.4)' }} />
        </div>
        <div className="text-center space-y-1.5">
          <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
            {isDragOver ? 'Release to decode' : 'Drop an audio file here'}
          </p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>
            MP3, WAV, OGG, FLAC — up to 25MB
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onBrowse}
            className="px-4 py-2 text-xs font-medium rounded-[var(--radius-button)] transition-all duration-200 cursor-pointer"
            style={{
              background: 'rgba(var(--accent-rgb), 0.1)',
              color: 'var(--accent)',
              border: '1px solid rgba(var(--accent-rgb), 0.2)',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(var(--accent-rgb), 0.18)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(var(--accent-rgb), 0.1)'; }}
          >
            Browse files
          </button>
          <button
            onClick={handleSample}
            className="px-4 py-2 text-xs font-medium rounded-[var(--radius-button)] transition-all duration-200 cursor-pointer"
            style={{
              background: 'rgba(255,255,255,0.04)',
              color: 'rgba(255,255,255,0.5)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
          >
            Try a sample
          </button>
        </div>
      </div>
    ),
    'hover-armed': null,
    'drag-over': null,
    decoding: (
      <div className="flex flex-col items-center gap-4">
        <Loader size="md" text="Decoding audio..." />
        {fileName && (
          <p className="text-xs font-mono" style={{ color: 'rgba(255,255,255,0.4)' }}>
            {fileName} ({formatFileSize(fileSize)})
          </p>
        )}
      </div>
    ),
    analyzing: (
      <div className="flex flex-col items-center gap-4">
        <Loader size="md" text="Analyzing sections..." />
      </div>
    ),
    success: (
      <motion.div
        className="flex items-center gap-3"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'rgba(var(--accent-rgb), 0.12)' }}
        >
          <FileAudio size={16} strokeWidth={1.5} style={{ color: 'var(--accent)' }} />
        </div>
        <div>
          <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>{fileName}</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{formatFileSize(fileSize)}</p>
        </div>
        <button
          onClick={clearFile}
          className="ml-auto text-xs px-3 py-1 rounded-[var(--radius-tag)] cursor-pointer transition-colors duration-200"
          style={{ color: 'rgba(255,255,255,0.4)', background: 'rgba(255,255,255,0.04)' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--foreground)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.4)'; }}
        >
          Change
        </button>
      </motion.div>
    ),
    error: (
      <div className="flex flex-col items-center gap-3 text-center">
        <Music size={24} strokeWidth={1.5} style={{ color: '#FF2D7A' }} />
        <p className="text-sm" style={{ color: '#FF2D7A' }}>
          {decodeError || 'Failed to decode audio'}
        </p>
        <button
          onClick={clearFile}
          className="text-xs px-4 py-1.5 rounded-[var(--radius-button)] cursor-pointer"
          style={{
            background: 'rgba(255,45,122,0.1)',
            color: '#FF2D7A',
            border: '1px solid rgba(255,45,122,0.2)',
          }}
        >
          Try another file
        </button>
      </div>
    ),
  };

  const isCompact = dropzoneState === 'success';

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={onFileChange}
        aria-label="Upload audio file"
      />
      <AnimatePresence mode="wait">
        <motion.div
          key={dropzoneState}
          className={`glass ${isCompact ? 'p-3' : 'p-8'} ${isDragOver ? '' : ''}`}
          style={{
            borderColor: isDragOver ? 'rgba(var(--accent-rgb), 0.3)' : undefined,
            cursor: dropzoneState === 'idle' ? 'pointer' : 'default',
          }}
          onDragEnter={onDragEnter}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={dropzoneState === 'idle' ? onBrowse : undefined}
          initial={{ opacity: 0, height: isCompact ? 56 : 200 }}
          animate={{ opacity: 1, height: isCompact ? 56 : 200 }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          role="region"
          aria-label="Audio file upload"
        >
          <div className="flex items-center justify-center h-full">
            {stateMap[dropzoneState]}
          </div>
        </motion.div>
      </AnimatePresence>
    </>
  );
}
