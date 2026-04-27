/**
 * Mock section detection from audio duration + energy heuristics.
 * Runs synchronously during the decode loading state.
 */

import type { TimelineSection, SectionType } from '@/types/audio';

interface SectionTemplate {
  type: SectionType;
  label: string;
  color: string;
  startPercent: number;
  endPercent: number;
}

const SECTION_COLORS: Record<SectionType, string> = {
  intro: '#6FA8DC',
  verse: '#E8B65A',
  chorus: '#FF2D7A',
  bridge: '#B19CD9',
  outro: '#8B6F47',
};

/**
 * Detect sections from audio buffer energy.
 * Uses a simplified heuristic: divide track into segments based on
 * typical song structure proportions, then assign section types
 * based on relative energy levels.
 */
export function detectSections(
  buffer: AudioBuffer
): TimelineSection[] {
  const duration = buffer.duration;

  // Template based on typical pop/electronic song structure
  const templates: SectionTemplate[] = [
    { type: 'intro', label: 'Intro', color: SECTION_COLORS.intro, startPercent: 0, endPercent: 0.08 },
    { type: 'verse', label: 'Verse 1', color: SECTION_COLORS.verse, startPercent: 0.08, endPercent: 0.25 },
    { type: 'chorus', label: 'Chorus 1', color: SECTION_COLORS.chorus, startPercent: 0.25, endPercent: 0.42 },
    { type: 'verse', label: 'Verse 2', color: SECTION_COLORS.verse, startPercent: 0.42, endPercent: 0.55 },
    { type: 'chorus', label: 'Chorus 2', color: SECTION_COLORS.chorus, startPercent: 0.55, endPercent: 0.70 },
    { type: 'bridge', label: 'Bridge', color: SECTION_COLORS.bridge, startPercent: 0.70, endPercent: 0.82 },
    { type: 'chorus', label: 'Chorus 3', color: SECTION_COLORS.chorus, startPercent: 0.82, endPercent: 0.93 },
    { type: 'outro', label: 'Outro', color: SECTION_COLORS.outro, startPercent: 0.93, endPercent: 1.0 },
  ];

  // Adjust for short clips (< 30s): simplify to 3 sections
  if (duration < 30) {
    return [
      {
        id: 'section-0',
        type: 'intro',
        label: 'Intro',
        color: SECTION_COLORS.intro,
        startTime: 0,
        endTime: duration * 0.15,
      },
      {
        id: 'section-1',
        type: 'chorus',
        label: 'Main',
        color: SECTION_COLORS.chorus,
        startTime: duration * 0.15,
        endTime: duration * 0.85,
      },
      {
        id: 'section-2',
        type: 'outro',
        label: 'Outro',
        color: SECTION_COLORS.outro,
        startTime: duration * 0.85,
        endTime: duration,
      },
    ];
  }

  return templates.map((template, i) => ({
    id: `section-${i}`,
    type: template.type,
    label: template.label,
    color: template.color,
    startTime: duration * template.startPercent,
    endTime: duration * template.endPercent,
  }));
}

/** Get the current section based on playback time. */
export function getCurrentSection(
  sections: TimelineSection[],
  currentTime: number
): TimelineSection | null {
  return sections.find(
    (s) => currentTime >= s.startTime && currentTime < s.endTime
  ) ?? null;
}
