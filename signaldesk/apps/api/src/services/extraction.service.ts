import fs from 'fs/promises';
import pdf from 'pdf-parse';

export async function extractTextFromPDF(filePath: string): Promise<string> {
  const dataBuffer = await fs.readFile(filePath);
  const data = await pdf(dataBuffer);
  return data.text;
}

export async function extractTextFromFile(filePath: string, mimeType: string): Promise<string> {
  if (mimeType === 'application/pdf') {
    return extractTextFromPDF(filePath);
  } else if (mimeType === 'text/plain' || mimeType === 'text/markdown') {
    return fs.readFile(filePath, 'utf-8');
  } else {
    throw new Error(`Unsupported MIME type: ${mimeType}`);
  }
}
