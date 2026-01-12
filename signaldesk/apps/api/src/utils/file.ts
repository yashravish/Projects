import path from 'path';

/**
 * Sanitizes a filename to prevent path traversal attacks
 * Removes directory separators and other potentially dangerous characters
 */
export function sanitizeFilename(filename: string): string {
  // Remove any path components
  const basename = path.basename(filename);

  // Remove or replace potentially dangerous characters
  // Allow alphanumeric, dots, hyphens, and underscores only
  const sanitized = basename.replace(/[^a-zA-Z0-9._-]/g, '_');

  // Prevent hidden files on Unix systems
  if (sanitized.startsWith('.')) {
    return '_' + sanitized.substring(1);
  }

  // Ensure the filename is not empty after sanitization
  if (!sanitized || sanitized === '_') {
    return 'unnamed_file';
  }

  return sanitized;
}
