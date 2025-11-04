import { NextRequest } from "next/server";
import OpenAI from "openai";
import { requireEnv } from "@/lib/env";
import { z } from "zod";
import { ALLOWED_MODELS } from "@/lib/models";
import { auth } from "@/app/api/auth/[...nextauth]/route";
import { checkRateLimit } from "@/lib/ratelimit";
import { sanitizePrompt } from "@/lib/validation";
import { logError } from "@/lib/errors";
import { LIMITS } from "@/lib/constants";

export const runtime = "nodejs";

const generateSchema = z.object({
  model: z.enum(Object.keys(ALLOWED_MODELS) as [string, ...string[]]),
  prompt: z.string().min(1).max(LIMITS.PROMPT_MAX_LENGTH),
  temperature: z.number().min(0).max(2).default(0.7),
  top_p: z.number().min(0).max(1).default(1),
  max_tokens: z.number().min(1).max(LIMITS.MAX_TOKENS_LIMIT).default(LIMITS.MAX_TOKENS_DEFAULT)
});

export async function POST(req: NextRequest) {
  try {
    // 1. Authentication check
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    // 2. Rate limiting
    const ip = req.headers.get("x-forwarded-for") ?? req.headers.get("x-real-ip") ?? "127.0.0.1";
    const identifier = `ratelimit:${session.user?.email || ip}`;
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

    // 3. Validate and sanitize input
    const body = await req.json();
    const { model, prompt: rawPrompt, temperature, top_p, max_tokens } = generateSchema.parse(body);
    const prompt = sanitizePrompt(rawPrompt);

    // Validate max_tokens against model-specific limit
    const modelConfig = ALLOWED_MODELS[model as keyof typeof ALLOWED_MODELS];
    if (max_tokens > modelConfig.max_out) {
      return Response.json(
        {
          error: `max_tokens (${max_tokens}) exceeds model limit (${modelConfig.max_out})`
        },
        { status: 400 }
      );
    }

    const apiKey = requireEnv("OPENAI_API_KEY");
    const client = new OpenAI({ apiKey });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          const completion = await client.chat.completions.create({
            model,
            messages: [{ role: "user", content: prompt }],
            temperature,
            top_p,
            max_tokens,
            stream: true
          });

          for await (const chunk of completion) {
            const token = chunk.choices?.[0]?.delta?.content ?? "";
            if (token) {
              controller.enqueue(encoder.encode(`data: ${token}\n\n`));
            }
          }
        } catch (e: any) {
          const safeMsg = e?.status === 400 ? e?.message : "Stream processing error";
          controller.enqueue(encoder.encode(`event: error\ndata: ${safeMsg}\n\n`));
          logError(e, { context: "streaming", model, prompt: prompt.slice(0, 100) });
        } finally {
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no"
      }
    });
  } catch (error: any) {
    logError(error, { endpoint: "/api/generate" });

    // Handle Zod validation errors
    if (error.name === "ZodError") {
      return Response.json(
        { error: "Invalid request", details: error.errors },
        { status: 400 }
      );
    }

    // Sanitize error messages for client safety
    // OpenAI SDK errors with status 400 or type "invalid_request_error" are safe to show
    const isClientError =
      error?.status === 400 ||
      error?.type === "invalid_request_error" ||
      error?.code === "invalid_api_key";

    const message = isClientError
      ? (error?.message || "Invalid request")
      : "Failed to generate response";

    const status = isClientError ? 400 : 500;

    return Response.json({ error: message }, { status });
  }
}
