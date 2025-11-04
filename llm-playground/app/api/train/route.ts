import { NextRequest } from "next/server";
import { auth } from "@/lib/auth-helpers";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    // Add authentication check
    const session = await auth();
    if (!session) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { text } = await req.json();

    const tokens = String(text ?? "")
      .split(/\s+/)
      .filter((t: string) => t.trim().length);

    if (tokens.length < 50) {
      return Response.json(
        { error: "Provide more text (≥ 50 tokens)" },
        { status: 400 }
      );
    }

    const epochs = 6;
    const losses = Array.from({ length: epochs }, (_, i) => 2.0 * Math.exp(-i * 0.6));

    return Response.json({
      epochs,
      losses,
      checkpoint: "toy-openai-sft-stub"
    });
  } catch (error) {
    console.error("Train API error:", error);
    return Response.json({ error: "Invalid request" }, { status: 400 });
  }
}
