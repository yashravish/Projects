/**
 * Web scraping and citation utilities
 * Provides helpers for fetching, sanitizing, and formatting web content with citations
 */

import { WEB_LIMITS } from './constants';

export interface CitationSource {
  idx: number;
  url: string;
  title: string;
}

/**
 * Sanitizes and trims HTML content to a specified character limit
 * - Strips HTML tags (preserving line breaks)
 * - Removes scripts, styles, and other non-content elements
 * - Collapses whitespace
 * - Truncates to character cap
 */
export function sanitizeAndTrimHtml(html: string, perPageCap: number): string {
  if (!html) return '';

  let text = html;

  // Remove script and style tags with their content
  text = text.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
  text = text.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');

  // Remove HTML comments
  text = text.replace(/<!--[\s\S]*?-->/g, '');

  // Convert common block elements to line breaks
  text = text.replace(/<(br|p|div|h[1-6]|li|tr)[^>]*>/gi, '\n');

  // Remove all remaining HTML tags
  text = text.replace(/<[^>]+>/g, ' ');

  // Decode common HTML entities
  text = text
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'");

  // Collapse multiple whitespace characters into single space
  text = text.replace(/[ \t]+/g, ' ');

  // Collapse multiple newlines into at most 2
  text = text.replace(/\n{3,}/g, '\n\n');

  // Trim whitespace from each line
  text = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join('\n');

  // Trim to character cap
  if (text.length > perPageCap) {
    text = text.substring(0, perPageCap) + '...';
  }

  return text.trim();
}

/**
 * Chooses top unique URLs from a list
 * - Normalizes URLs (removes protocol differences, www., trailing slashes, hashes)
 * - Deduplicates near-identical URLs
 * - Returns up to `limit` unique URLs
 */
export function chooseTopUniqueUrls(urls: string[], limit: number): string[] {
  if (!urls || urls.length === 0) return [];

  const seen = new Set<string>();
  const uniqueUrls: string[] = [];

  for (const url of urls) {
    if (uniqueUrls.length >= limit) break;

    try {
      // Normalize URL for comparison
      const normalized = normalizeUrl(url);

      if (!seen.has(normalized)) {
        seen.add(normalized);
        uniqueUrls.push(url); // Keep original URL
      }
    } catch {
      // Skip invalid URLs
      continue;
    }
  }

  return uniqueUrls;
}

/**
 * Normalizes a URL for duplicate detection
 * - Converts to lowercase
 * - Removes protocol (http/https)
 * - Removes www. prefix
 * - Removes trailing slashes
 * - Removes hash fragments
 * - Removes common tracking parameters
 */
function normalizeUrl(url: string): string {
  let normalized = url.toLowerCase().trim();

  // Remove protocol
  normalized = normalized.replace(/^https?:\/\//, '');

  // Remove www.
  normalized = normalized.replace(/^www\./, '');

  // Remove trailing slash
  normalized = normalized.replace(/\/$/, '');

  // Remove hash fragments
  normalized = normalized.replace(/#.*$/, '');

  // Remove common tracking parameters
  const urlObj = new URL(url);
  const trackingParams = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'fbclid', 'gclid'];
  trackingParams.forEach((param) => {
    urlObj.searchParams.delete(param);
  });

  // Reconstruct without protocol and www
  normalized = urlObj.hostname.replace(/^www\./, '') + urlObj.pathname + urlObj.search;
  normalized = normalized.replace(/\/$/, '');

  return normalized;
}

/**
 * Builds a prompt that instructs the LLM to summarize content with citations
 * The prompt includes:
 * - User's original query
 * - Scraped content from multiple sources
 * - Instructions to cite using [1], [2], etc.
 * - List of sources with URLs and titles
 *
 * IMPORTANT: Ensures the final prompt doesn't exceed maxLength (default 15000 chars)
 * to leave room for the API's 16k limit
 */
export function buildCitedPrompt(
  query: string,
  snippets: string[],
  sources: CitationSource[],
  maxLength: number = 15000
): string {
  if (!snippets || snippets.length === 0) {
    return `User query: ${query}\n\nNo web sources were found. Please provide a helpful response based on your general knowledge, but note that you don't have access to current web information for this query.`;
  }

  const instructions = `You are a helpful assistant that summarizes web content with citations.

User Query: ${query}

## Content from Web Sources

`;

  const footer = `

Instructions:
- Provide a comprehensive answer to the user's query based on the sources above
- Use inline citations like [1], [2], etc. to reference the sources
- Cite sources whenever you make a specific claim or use information from them
- If sources contradict each other, mention both perspectives with their citations
- Write in a clear, concise manner
- If the sources don't fully answer the query, note what information is missing
- Do not make up information not present in the sources

Please provide your answer now:`;

  // Build the sources section
  let sourcesSection = '\n\n## Sources\n';
  sources.forEach((source) => {
    sourcesSection += `[${source.idx}] ${source.title}\n    ${source.url}\n`;
  });

  // Calculate how much space we have for content
  const overhead = instructions.length + footer.length + sourcesSection.length;
  const availableSpace = maxLength - overhead - 500; // Extra buffer for safety

  if (availableSpace <= 0) {
    throw new Error('Query and sources metadata too large. Please use fewer sources or a shorter query.');
  }

  // Build the content section, truncating if necessary
  let contentSection = '';
  let currentLength = 0;
  let includedSnippets = 0;

  for (let index = 0; index < snippets.length; index++) {
    const sourceIdx = sources[index]?.idx || index + 1;
    const header = `### Source [${sourceIdx}]: ${sources[index]?.title || 'Untitled'}\n`;
    const snippet = snippets[index];
    const sectionLength = header.length + snippet.length + 2; // +2 for \n\n

    if (currentLength + sectionLength <= availableSpace) {
      contentSection += header + snippet + '\n\n';
      currentLength += sectionLength;
      includedSnippets++;
    } else {
      // Try to fit a truncated version of this snippet
      const remaining = availableSpace - currentLength - header.length;
      if (remaining > 200) {
        // Only include if we can fit at least 200 chars
        const truncated = snippet.substring(0, remaining - 3) + '...';
        contentSection += header + truncated + '\n\n';
        includedSnippets++;
      }
      break; // Stop adding more snippets
    }
  }

  // Update sources list to only include what we actually used
  if (includedSnippets < sources.length) {
    sourcesSection = '\n\n## Sources\n';
    sources.slice(0, includedSnippets).forEach((source) => {
      sourcesSection += `[${source.idx}] ${source.title}\n    ${source.url}\n`;
    });
  }

  const prompt = instructions + contentSection + sourcesSection + footer;

  // Final safety check
  if (prompt.length > maxLength) {
    throw new Error('Prompt too large even after truncation. Please try a shorter query.');
  }

  return prompt;
}

/**
 * Aggregates multiple snippets and ensures total context doesn't exceed the cap
 * Returns truncated snippets array if needed
 */
export function aggregateSnippets(
  snippets: string[],
  totalCap: number = WEB_LIMITS.WEB_TOTAL_CONTEXT_CHAR_CAP
): string[] {
  if (!snippets || snippets.length === 0) return [];

  let totalChars = 0;
  const result: string[] = [];

  for (const snippet of snippets) {
    if (totalChars + snippet.length <= totalCap) {
      result.push(snippet);
      totalChars += snippet.length;
    } else {
      // Add partial snippet if there's room
      const remaining = totalCap - totalChars;
      if (remaining > 100) {
        // Only add if at least 100 chars remain
        result.push(snippet.substring(0, remaining) + '...');
      }
      break;
    }
  }

  return result;
}
