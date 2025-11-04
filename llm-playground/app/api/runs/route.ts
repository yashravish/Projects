import { prisma } from "@/lib/prisma";
import { auth } from "@/lib/auth-helpers";
import { NextRequest } from "next/server";
import { z } from "zod";
import { LIMITS } from "@/lib/constants";
import { logError } from "@/lib/errors";

export const runtime = "nodejs";

export async function GET() {
  try {
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const rows = await prisma.run.findMany({
      take: LIMITS.RUNS_TABLE_LIMIT,
      orderBy: { created_at: "desc" }
    });

    return Response.json(rows);
  } catch (error) {
    logError(error as Error, { endpoint: "GET /api/runs" });
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}

const runCreateSchema = z.object({
  model: z.string().min(1),
  prompt: z.string().min(1),
  params: z.string(),
  tokensInput: z.number().int().min(0),
  tokensOutput: z.number().int().min(0),
  latencyMs: z.number().int().min(0),
  costUsd: z.number().min(0),
  outputText: z.string().optional().default(""),
  replayFrames: z.string().optional(), // JSON array of [ms, token]
  isPublic: z.boolean().optional().default(false)
});

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const validated = runCreateSchema.parse(body);

    const row = await prisma.run.create({
      data: {
        model: validated.model,
        prompt: validated.prompt,
        params: validated.params,
        tokens_input: validated.tokensInput,
        tokens_output: validated.tokensOutput,
        latency_ms: validated.latencyMs,
        cost_usd: validated.costUsd,
        output_text: validated.outputText,
        replay_frames: validated.replayFrames,
        is_public: validated.isPublic
      }
    });

    return Response.json(row, { status: 201 });
  } catch (error: any) {
    logError(error, { endpoint: "POST /api/runs" });

    if (error.name === "ZodError") {
      return Response.json(
        { error: "Invalid payload", details: error.errors },
        { status: 400 }
      );
    }

    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function DELETE() {
  try {
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const result = await prisma.run.deleteMany({});

    return Response.json(
      { message: "All runs deleted", count: result.count },
      { status: 200 }
    );
  } catch (error) {
    logError(error as Error, { endpoint: "DELETE /api/runs" });
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
