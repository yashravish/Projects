/**
 * Lightweight keyword-based retrieval system for RAG
 * Implements chunking, tokenization, and TF-IDF scoring
 */

import { SUPPORT_LIMITS } from './constants';

/**
 * Common English stopwords to filter out
 */
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
  'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
  'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or', 'not',
]);

/**
 * Chunks text into smaller pieces with overlap
 * Strategy:
 * 1. Split by paragraphs (double newlines)
 * 2. Merge/split to target size with overlap
 */
export function chunkText(
  text: string,
  targetSize: number = SUPPORT_LIMITS.CHUNK_TARGET_SIZE,
  overlap: number = SUPPORT_LIMITS.CHUNK_OVERLAP
): string[] {
  if (!text || text.trim().length === 0) {
    return [];
  }

  // Split into paragraphs
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  if (paragraphs.length === 0) {
    return [];
  }

  const chunks: string[] = [];
  let currentChunk = '';

  for (const paragraph of paragraphs) {
    // If adding this paragraph would exceed target size
    if (currentChunk.length > 0 && currentChunk.length + paragraph.length + 2 > targetSize) {
      // Save current chunk
      chunks.push(currentChunk);

      // Start new chunk with overlap from previous chunk
      const words = currentChunk.split(/\s+/);
      const overlapWords = words.slice(-Math.floor(overlap / 6)); // Roughly overlap chars / avg word length
      currentChunk = overlapWords.join(' ') + '\n\n' + paragraph;
    } else {
      // Add paragraph to current chunk
      if (currentChunk.length > 0) {
        currentChunk += '\n\n' + paragraph;
      } else {
        currentChunk = paragraph;
      }
    }
  }

  // Don't forget the last chunk
  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  // Handle very long single paragraphs by splitting them
  const finalChunks: string[] = [];
  for (const chunk of chunks) {
    if (chunk.length <= targetSize * 1.5) {
      // Chunk is reasonable size
      finalChunks.push(chunk);
    } else {
      // Split long chunk into sentences or by size
      const sentences = chunk.split(/[.!?]+\s+/);
      let subChunk = '';

      for (const sentence of sentences) {
        if (subChunk.length > 0 && subChunk.length + sentence.length > targetSize) {
          finalChunks.push(subChunk);

          // Start new subchunk with overlap
          const words = subChunk.split(/\s+/);
          const overlapWords = words.slice(-Math.floor(overlap / 6));
          subChunk = overlapWords.join(' ') + ' ' + sentence;
        } else {
          subChunk += (subChunk.length > 0 ? '. ' : '') + sentence;
        }
      }

      if (subChunk.length > 0) {
        finalChunks.push(subChunk);
      }
    }
  }

  return finalChunks;
}

/**
 * Tokenizes text: lowercase, strip punctuation, split on whitespace, remove stopwords
 */
export function tokenizeLowerAscii(text: string): string[] {
  if (!text) return [];

  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ') // Replace punctuation with space
    .split(/\s+/) // Split on whitespace
    .filter((token) => token.length > 0 && !STOPWORDS.has(token));
}

/**
 * Computes term frequency for a list of tokens
 */
function computeTermFrequency(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();

  for (const token of tokens) {
    tf.set(token, (tf.get(token) || 0) + 1);
  }

  // Normalize by total tokens
  const total = tokens.length || 1;
  for (const [term, count] of tf.entries()) {
    tf.set(term, count / total);
  }

  return tf;
}

/**
 * Computes inverse document frequency across all chunks
 */
function computeIDF(chunks: string[]): Map<string, number> {
  const idf = new Map<string, number>();
  const documentCount = chunks.length;

  // Count how many chunks contain each term
  const termDocCount = new Map<string, number>();

  for (const chunk of chunks) {
    const tokens = new Set(tokenizeLowerAscii(chunk)); // Use Set to count unique terms per chunk
    for (const token of tokens) {
      termDocCount.set(token, (termDocCount.get(token) || 0) + 1);
    }
  }

  // Compute IDF: log(total docs / docs containing term)
  for (const [term, docCount] of termDocCount.entries()) {
    idf.set(term, Math.log((documentCount + 1) / (docCount + 1))); // +1 for smoothing
  }

  return idf;
}

/**
 * Scores chunks based on TF-IDF similarity to query
 * Returns array of { i: chunkIndex, score: relevanceScore }
 */
export function scoreChunks(
  query: string,
  chunks: string[]
): Array<{ i: number; score: number }> {
  if (!query || !chunks || chunks.length === 0) {
    return [];
  }

  const queryTokens = tokenizeLowerAscii(query);
  if (queryTokens.length === 0) {
    return chunks.map((_, i) => ({ i, score: 0 }));
  }

  const queryTF = computeTermFrequency(queryTokens);
  const idf = computeIDF(chunks);

  const scores: Array<{ i: number; score: number }> = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunkTokens = tokenizeLowerAscii(chunks[i]);
    const chunkTF = computeTermFrequency(chunkTokens);

    let score = 0;

    // TF-IDF scoring: sum of (query_tf * chunk_tf * idf) for matching terms
    for (const [term, qTF] of queryTF.entries()) {
      if (chunkTF.has(term)) {
        const cTF = chunkTF.get(term) || 0;
        const termIDF = idf.get(term) || 0;
        score += qTF * cTF * termIDF;
      }
    }

    // Optional: Boost for exact phrase matches
    const queryLower = query.toLowerCase();
    const chunkLower = chunks[i].toLowerCase();
    if (chunkLower.includes(queryLower)) {
      score *= 1.5; // 50% boost for exact phrase match
    }

    // Optional: Boost for query terms appearing close together
    const positions: number[][] = [];
    for (const term of queryTokens) {
      const termPositions: number[] = [];
      let idx = 0;
      while ((idx = chunkLower.indexOf(term, idx)) !== -1) {
        termPositions.push(idx);
        idx += term.length;
      }
      if (termPositions.length > 0) {
        positions.push(termPositions);
      }
    }

    // If multiple query terms appear close together, boost score
    if (positions.length > 1) {
      for (let j = 0; j < positions[0].length; j++) {
        const firstPos = positions[0][j];
        let allClose = true;
        for (let k = 1; k < positions.length; k++) {
          const closestPos = positions[k].reduce((prev, curr) =>
            Math.abs(curr - firstPos) < Math.abs(prev - firstPos) ? curr : prev
          );
          if (Math.abs(closestPos - firstPos) > 100) {
            // Terms more than 100 chars apart
            allClose = false;
            break;
          }
        }
        if (allClose) {
          score *= 1.2; // 20% boost for proximity
          break;
        }
      }
    }

    scores.push({ i, score });
  }

  return scores;
}

/**
 * Selects top K chunks based on scores
 * Returns indices of selected chunks
 */
export function selectTopK(
  scores: Array<{ i: number; score: number }>,
  K: number = SUPPORT_LIMITS.TOP_K_CHUNKS,
  minScore: number = SUPPORT_LIMITS.MIN_SCORE_THRESHOLD
): number[] {
  return scores
    .filter((s) => s.score >= minScore) // Filter by minimum score
    .sort((a, b) => b.score - a.score) // Sort descending by score
    .slice(0, K) // Take top K
    .map((s) => s.i); // Extract indices
}

/**
 * One-stop retrieval function
 * Takes a query and text, returns top K relevant chunks
 */
export function retrieve(
  query: string,
  text: string,
  K: number = SUPPORT_LIMITS.TOP_K_CHUNKS
): { indices: number[]; chunks: string[] } {
  if (!query || !text) {
    return { indices: [], chunks: [] };
  }

  // Step 1: Chunk the text
  const chunks = chunkText(text);

  if (chunks.length === 0) {
    return { indices: [], chunks: [] };
  }

  // Step 2: Score chunks
  const scores = scoreChunks(query, chunks);

  // Step 3: Select top K
  const indices = selectTopK(scores, K);

  // Step 4: Return selected chunks in order of relevance
  const selectedChunks = indices.map((i) => chunks[i]);

  return { indices, chunks: selectedChunks };
}
