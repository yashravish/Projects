export class AppError extends Error {
  constructor(
    public message: string,
    public status: number = 500,
    public code?: string
  ) {
    super(message);
    this.name = "AppError";
  }
}

export function handleApiError(error: unknown): Response {
  if (error instanceof AppError) {
    return Response.json(
      { error: error.message, code: error.code },
      { status: error.status }
    );
  }

  if ((error as any).name === "ZodError") {
    return Response.json(
      { error: "Validation failed", details: (error as any).errors },
      { status: 400 }
    );
  }

  console.error("Unhandled error:", error);
  return Response.json({ error: "Internal server error" }, { status: 500 });
}

export function logError(error: Error | unknown, context?: Record<string, any>) {
  console.error(error, context);

  // In production, you would send this to Sentry or similar
  // if (process.env.NODE_ENV === "production") {
  //   Sentry.captureException(error, { extra: context });
  // }
}
