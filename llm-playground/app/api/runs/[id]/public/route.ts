import { NextRequest } from "next/server";
import { prisma } from "@/lib/prisma";
import { auth } from "@/lib/auth-helpers";

export async function PUT(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const session = await auth();
  if (!session) return Response.json({ error: "Unauthorized" }, { status: 401 });

  const { isPublic } = await req.json().catch(() => ({ isPublic: true }));
  const { id } = await params;

  const row = await prisma.run.update({
    where: { id },
    data: { is_public: Boolean(isPublic) }
  });

  return Response.json({ id: row.id, isPublic: row.is_public });
}


