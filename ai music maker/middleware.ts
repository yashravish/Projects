import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

/**
 * Middleware that runs on Edge Runtime.
 * Supabase session management moved to individual server components/route handlers
 * to avoid Edge Runtime compatibility issues.
 */
export function middleware(_request: NextRequest) {
  // For now, just pass through all requests
  // Auth checks are handled in server components and API routes
  return NextResponse.next()
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - images - .svg, .png, .jpg, .jpeg, .gif, .webp
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
}
