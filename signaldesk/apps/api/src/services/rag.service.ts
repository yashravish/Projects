import OpenAI from 'openai';
import { CONFIG } from '@signaldesk/shared';
import { config } from '../config';
import { prisma } from '../db';
import { generateEmbedding } from './embedding.service';

let openai: OpenAI | null = null;

if (config.openaiApiKey) {
  openai = new OpenAI({
    apiKey: config.openaiApiKey,
  });
}

interface RetrievedChunk {
  id: string;
  content: string;
  documentId: string;
  chunkIndex: number;
  relevanceScore: number;
}

interface RAGResponse {
  answer: string;
  citations: Array<{
    chunkId: string;
    documentId: string;
    documentName: string;
    chunkIndex: number;
    content: string;
    relevanceScore: number;
  }>;
  keyFacts: Array<{
    fact: string;
    source: string;
  }>;
  summary: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export async function performRAGQuery(
  question: string,
  collectionId: string,
  userId: string
): Promise<RAGResponse> {
  const questionEmbedding = await generateEmbedding(question);

  let retrievedChunks: RetrievedChunk[];

  if (questionEmbedding) {
    const embeddingString = `[${questionEmbedding.join(',')}]`;

    const chunks = await prisma.$queryRaw<
      Array<{
        id: string;
        content: string;
        documentId: string;
        chunkIndex: number;
        score: number;
      }>
    >`
      SELECT
        dc.id,
        dc.content,
        dc."documentId",
        dc."chunkIndex",
        1 - (dc.embedding <=> ${embeddingString}::vector) AS score
      FROM "DocumentChunk" dc
      INNER JOIN "Document" d ON dc."documentId" = d.id
      WHERE dc."userId" = ${userId}
        AND d."collectionId" = ${collectionId}
        AND dc.embedding IS NOT NULL
      ORDER BY dc.embedding <=> ${embeddingString}::vector
      LIMIT ${CONFIG.RETRIEVAL_TOP_K}
    `;

    retrievedChunks = chunks.map((chunk: typeof chunks[0]) => ({
      id: chunk.id,
      content: chunk.content,
      documentId: chunk.documentId,
      chunkIndex: chunk.chunkIndex,
      relevanceScore: chunk.score,
    }));
  } else {
    const chunks = await prisma.documentChunk.findMany({
      where: {
        userId,
        document: {
          collectionId,
        },
      },
      take: CONFIG.RETRIEVAL_TOP_K,
      orderBy: {
        chunkIndex: 'asc',
      },
      select: {
        id: true,
        content: true,
        documentId: true,
        chunkIndex: true,
      },
    });

    retrievedChunks = chunks.map((chunk: typeof chunks[0]) => ({
      ...chunk,
      relevanceScore: 0,
    }));
  }

  const documents = await prisma.document.findMany({
    where: {
      id: {
        in: retrievedChunks.map((chunk) => chunk.documentId),
      },
    },
    select: {
      id: true,
      originalName: true,
    },
  });

  const documentMap = new Map(documents.map((doc: typeof documents[0]) => [doc.id, doc.originalName]));

  const citations = retrievedChunks.map((chunk) => ({
    chunkId: chunk.id,
    documentId: chunk.documentId,
    documentName: (documentMap.get(chunk.documentId) || 'Unknown') as string,
    chunkIndex: chunk.chunkIndex,
    content: chunk.content,
    relevanceScore: chunk.relevanceScore,
  }));

  if (!openai || !config.openaiApiKey) {
    return {
      answer:
        'This is a stub response. Configure OPENAI_API_KEY for real AI responses. Retrieved chunks are shown in citations.',
      citations,
      keyFacts: [{ fact: 'Stub mode active - No OpenAI API key configured', source: 'system' }],
      summary: 'Stub mode: No AI processing available.',
      usage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
    };
  }

  const contextChunks = retrievedChunks
    .map(
      (chunk, index) =>
        `[${index + 1}] (Document: ${documentMap.get(chunk.documentId)}, Chunk ${chunk.chunkIndex + 1}):\n${chunk.content}`
    )
    .join('\n\n');

  const systemPrompt = `You are a document analysis assistant. Answer the user's question based ONLY on the provided context.

RULES:
1. Only use information from the provided context
2. Cite sources using [1], [2], etc. corresponding to chunk numbers
3. If the context doesn't contain enough information, say so explicitly
4. Extract 3-5 key facts from the context relevant to the question
5. Provide a brief executive summary (max 150 words)

CONTEXT:
${contextChunks}

Respond in this exact JSON format:
{
  "answer": "Your detailed answer with [1], [2] citations...",
  "keyFacts": [
    {"fact": "Key fact 1", "source": "chunk-id"},
    {"fact": "Key fact 2", "source": "chunk-id"}
  ],
  "summary": "Executive summary in under 150 words..."
}`;

  try {
    const completion = await openai.chat.completions.create({
      model: CONFIG.CHAT_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question },
      ],
      temperature: CONFIG.CHAT_TEMPERATURE,
      max_tokens: CONFIG.CHAT_MAX_TOKENS,
      response_format: { type: 'json_object' },
    });

    const response = JSON.parse(completion.choices[0].message.content || '{}');

    const keyFactsWithSources = (response.keyFacts || []).map(
      (fact: { fact: string; source?: string }, index: number) => ({
        fact: fact.fact,
        source: fact.source || retrievedChunks[index]?.id || 'unknown',
      })
    );

    return {
      answer: response.answer || 'No answer generated',
      citations,
      keyFacts: keyFactsWithSources,
      summary: response.summary || '',
      usage: {
        promptTokens: completion.usage?.prompt_tokens || 0,
        completionTokens: completion.usage?.completion_tokens || 0,
        totalTokens: completion.usage?.total_tokens || 0,
      },
    };
  } catch (error) {
    console.error('Error in RAG query:', error);
    throw error;
  }
}
