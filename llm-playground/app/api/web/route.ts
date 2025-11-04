import { NextRequest } from "next/server";
import { z } from "zod";
import { auth } from "@/app/api/auth/[...nextauth]/route";
import { checkRateLimit } from "@/lib/ratelimit";
import { logError } from "@/lib/errors";
import { LIMITS, WEB_LIMITS } from "@/lib/constants";
import { sanitizeAndTrimHtml, chooseTopUniqueUrls, aggregateSnippets } from "@/lib/web";

export const runtime = "nodejs";

const webSearchSchema = z.object({
  query: z.string().min(1).max(LIMITS.PROMPT_MAX_LENGTH),
  limit: z.number().min(1).max(WEB_LIMITS.WEB_MAX_RESULTS).optional(),
  seedUrls: z.array(z.string().url()).optional(), // For testing without search
});

interface FetchedSource {
  idx: number;
  url: string;
  title: string;
  text: string;
}

/**
 * Performs a DuckDuckGo HTML search and extracts result URLs
 * Returns an array of candidate URLs
 */
async function searchDuckDuckGo(query: string): Promise<string[]> {
  const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;

  try {
    const response = await fetch(searchUrl, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      },
      signal: AbortSignal.timeout(WEB_LIMITS.REQUEST_TIMEOUT_MS),
    });

    if (!response.ok) {
      throw new Error(`DuckDuckGo search failed: ${response.status}`);
    }

    const html = await response.text();

    // Extract URLs from DuckDuckGo result links
    // DDG uses format: <a class="result__url" href="/l/?uddg=...">
    // We need to extract the actual destination URL
    const urlMatches = html.matchAll(/class="result__url"[^>]*href="([^"]+)"/gi);
    const urls: string[] = [];

    for (const match of urlMatches) {
      const href = match[1];

      // Extract the uddg parameter which contains the encoded destination URL
      const uddgMatch = href.match(/uddg=([^&]+)/);
      if (uddgMatch) {
        try {
          const decodedUrl = decodeURIComponent(uddgMatch[1]);
          // Validate it's a proper URL
          if (decodedUrl.startsWith('http://') || decodedUrl.startsWith('https://')) {
            urls.push(decodedUrl);
          }
        } catch {
          // Skip invalid URLs
          continue;
        }
      }
    }

    // Fallback: also try to extract direct links from result snippets
    if (urls.length === 0) {
      const directMatches = html.matchAll(/class="result__a"[^>]*href="([^"]+)"/gi);
      for (const match of directMatches) {
        const url = match[1];
        if (url.startsWith('http://') || url.startsWith('https://')) {
          urls.push(url);
        }
      }
    }

    return urls.slice(0, 8); // Return top 8 candidates
  } catch (error: any) {
    logError(error, { context: "DuckDuckGo search", query });
    throw new Error("Failed to perform web search");
  }
}

/**
 * Fetches a single URL and extracts title and text
 */
async function fetchUrl(url: string, idx: number): Promise<FetchedSource | null> {
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      },
      signal: AbortSignal.timeout(WEB_LIMITS.REQUEST_TIMEOUT_MS),
      next: { revalidate: 60 }, // Cache for 60 seconds
    });

    if (!response.ok) {
      return null;
    }

    const html = await response.text();

    // Extract title
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    const title = titleMatch ? titleMatch[1].trim() : new URL(url).hostname;

    // Sanitize and trim HTML to text
    const text = sanitizeAndTrimHtml(html, WEB_LIMITS.WEB_PER_PAGE_CHAR_CAP);

    if (!text || text.length < 50) {
      // Skip sources with very little content
      return null;
    }

    return { idx, url, title, text };
  } catch (error: any) {
    // Silently skip failed fetches
    return null;
  }
}

/**
 * Fetches multiple URLs in parallel with bounded concurrency
 */
async function fetchUrlsInParallel(urls: string[], concurrency: number = 3): Promise<FetchedSource[]> {
  const results: FetchedSource[] = [];
  const queue = urls.map((url, idx) => ({ url, idx: idx + 1 }));

  // Process in batches for bounded concurrency
  for (let i = 0; i < queue.length; i += concurrency) {
    const batch = queue.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map(({ url, idx }) => fetchUrl(url, idx))
    );

    // Filter out nulls and add to results
    results.push(...batchResults.filter((r): r is FetchedSource => r !== null));
  }

  return results;
}

export async function POST(req: NextRequest) {
  try {
    // 1. Authentication check
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    // 2. Rate limiting
    const ip = req.headers.get("x-forwarded-for") ?? req.headers.get("x-real-ip") ?? "127.0.0.1";
    const identifier = `ratelimit:web:${session.user?.email || ip}`;
    const { success, remaining, reset } = checkRateLimit(identifier);

    if (!success) {
      return Response.json(
        { error: `Rate limit exceeded. Try again in ${Math.ceil((reset - Date.now()) / 1000)}s` },
        {
          status: 429,
          headers: {
            "X-RateLimit-Limit": "10",
            "X-RateLimit-Remaining": remaining.toString(),
            "X-RateLimit-Reset": new Date(reset).toISOString(),
          },
        }
      );
    }

    // 3. Validate input
    const body = await req.json();
    const { query, limit = WEB_LIMITS.WEB_MAX_RESULTS, seedUrls } = webSearchSchema.parse(body);

    if (!query.trim()) {
      return Response.json({ error: "Query cannot be empty" }, { status: 400 });
    }

    // 4. Get candidate URLs
    let candidateUrls: string[];

    if (seedUrls && seedUrls.length > 0) {
      // Use provided seed URLs (for testing)
      candidateUrls = seedUrls;
    } else {
      // Perform DuckDuckGo search
      candidateUrls = await searchDuckDuckGo(query);
    }

    if (candidateUrls.length === 0) {
      return Response.json(
        { error: "No search results found for this query" },
        { status: 404 }
      );
    }

    // 5. Select unique URLs
    const selectedUrls = chooseTopUniqueUrls(candidateUrls, limit);

    // 6. Fetch URLs in parallel
    const fetchedSources = await fetchUrlsInParallel(selectedUrls);

    if (fetchedSources.length === 0) {
      return Response.json(
        { error: "Failed to fetch any sources. Please try a different query." },
        { status: 500 }
      );
    }

    // 7. Aggregate snippets with total context cap
    const snippets = fetchedSources.map((s) => s.text);
    const aggregated = aggregateSnippets(snippets, WEB_LIMITS.WEB_TOTAL_CONTEXT_CHAR_CAP);

    // Build sources array (only include sources that made it into aggregated snippets)
    const sources = fetchedSources.slice(0, aggregated.length).map((s) => ({
      idx: s.idx,
      url: s.url,
      title: s.title,
    }));

    // Calculate total characters used
    const usedChars = aggregated.reduce((sum, snippet) => sum + snippet.length, 0);

    // 8. Return response
    return Response.json({
      query,
      sources,
      snippets: aggregated,
      usedChars,
    });
  } catch (error: any) {
    logError(error, { endpoint: "/api/web" });

    // Handle Zod validation errors
    if (error.name === "ZodError") {
      return Response.json(
        { error: "Invalid request", details: error.errors },
        { status: 400 }
      );
    }

    // Return sanitized error message
    const message = error.message || "Failed to fetch web sources";
    return Response.json({ error: message }, { status: 500 });
  }
}
