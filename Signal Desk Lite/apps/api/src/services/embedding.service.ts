import OpenAI from 'openai';
import { CONFIG } from '@signaldesk/shared';
import { config } from '../config';

let openai: OpenAI | null = null;

if (config.openaiApiKey) {
  openai = new OpenAI({
    apiKey: config.openaiApiKey,
  });
}

export async function generateEmbedding(text: string): Promise<number[] | null> {
  if (!openai) {
    return null;
  }

  try {
    const response = await openai.embeddings.create({
      model: CONFIG.EMBEDDING_MODEL,
      input: text,
    });

    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw error;
  }
}

export async function generateEmbeddings(texts: string[]): Promise<(number[] | null)[]> {
  if (!openai) {
    return texts.map(() => null);
  }

  try {
    const response = await openai.embeddings.create({
      model: CONFIG.EMBEDDING_MODEL,
      input: texts,
    });

    return response.data.map((item) => item.embedding);
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw error;
  }
}
