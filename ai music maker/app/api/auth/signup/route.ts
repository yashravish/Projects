import { NextResponse } from "next/server"
import { createClient as createSSRClient } from "@/lib/supabase/server"
import { SUPABASE_SERVICE_ROLE_KEY } from "@/lib/env"
import { createClient as createAdminClient } from "@supabase/supabase-js"
import { signUpSchema } from "@/lib/validation"
import { rateLimiter, RATE_LIMITS } from "@/lib/rate-limit"

export async function POST(request: Request) {
  try {
    // Get IP for rate limiting
    const ip = request.headers.get("x-forwarded-for") || request.headers.get("x-real-ip") || "anonymous"

    // Rate limit check
    const rateLimitResult = rateLimiter.check(`auth:${ip}`, RATE_LIMITS.AUTH.limit, RATE_LIMITS.AUTH.window)
    if (!rateLimitResult.success) {
      return NextResponse.json(
        { error: "Too many requests. Please try again later." },
        {
          status: 429,
          headers: {
            "X-RateLimit-Limit": RATE_LIMITS.AUTH.limit.toString(),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": new Date(rateLimitResult.reset).toISOString(),
          }
        }
      )
    }

    const body = await request.json().catch(() => ({}))

    // Validate input with Zod
    const validation = signUpSchema.safeParse(body)
    if (!validation.success) {
      return NextResponse.json(
        { error: validation.error.errors[0].message },
        { status: 400 }
      )
    }

    const { email, password } = validation.data

    if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
      return NextResponse.json({ error: "Server auth is not configured" }, { status: 500 })
    }

    // Create user as confirmed using the service role (no email confirmation)
    const admin = createAdminClient(process.env.NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    const { error: createError } = await admin.auth.admin.createUser({
      email,
      password,
      email_confirm: true,
    })
    if (createError) {
      return NextResponse.json({ error: createError.message }, { status: 400 })
    }

    // Sign the user in and set auth cookies on the response
    const supabase = await createSSRClient()
    const { error: signInError } = await supabase.auth.signInWithPassword({ email, password })
    if (signInError) {
      return NextResponse.json({ error: signInError.message }, { status: 400 })
    }

    return NextResponse.json({ ok: true })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unexpected error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}


