import { describe, it, expect, vi, beforeEach } from 'vitest';
import { extractTextFromFile } from '../services/extraction.service';
import fs from 'fs/promises';

// Mock fs/promises and pdf-parse
vi.mock('fs/promises');
vi.mock('pdf-parse', () => ({
  default: vi.fn((buffer) => Promise.resolve({ text: 'Extracted PDF text' })),
}));

describe('Extraction Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should extract text from PDF file', async () => {
    vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('fake pdf data'));

    const result = await extractTextFromFile('/fake/path.pdf', 'application/pdf');

    expect(result).toBe('Extracted PDF text');
    expect(fs.readFile).toHaveBeenCalledWith('/fake/path.pdf');
  });

  it('should extract text from plain text file', async () => {
    const textContent = 'This is plain text content';
    vi.mocked(fs.readFile).mockResolvedValue(textContent as any);

    const result = await extractTextFromFile('/fake/path.txt', 'text/plain');

    expect(result).toBe(textContent);
    expect(fs.readFile).toHaveBeenCalledWith('/fake/path.txt', 'utf-8');
  });

  it('should extract text from markdown file', async () => {
    const markdownContent = '# Markdown Title\n\nThis is markdown content';
    vi.mocked(fs.readFile).mockResolvedValue(markdownContent as any);

    const result = await extractTextFromFile('/fake/path.md', 'text/markdown');

    expect(result).toBe(markdownContent);
    expect(fs.readFile).toHaveBeenCalledWith('/fake/path.md', 'utf-8');
  });

  it('should throw error for unsupported MIME type', async () => {
    await expect(
      extractTextFromFile('/fake/path.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    ).rejects.toThrow('Unsupported MIME type');
  });

  it('should handle file read errors', async () => {
    vi.mocked(fs.readFile).mockRejectedValue(new Error('File not found'));

    await expect(extractTextFromFile('/fake/path.txt', 'text/plain')).rejects.toThrow('File not found');
  });
});
