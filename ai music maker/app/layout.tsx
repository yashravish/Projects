import type React from "react"
import type { Metadata } from "next"
import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Song Creator",
  description: "Create songs with AI"
}

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Refresh Supabase session on the server side
  const cookieStore = await cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options)
          )
        },
      },
    },
  )

  // Refresh the session to keep user logged in
  await supabase.auth.getUser()

  return (
    <html lang="en" className="dark">
      <body className="min-h-screen">
        <main className="min-h-screen mx-auto w-full max-w-6xl p-5 md:p-8">
          {children}
        </main>
      </body>
    </html>
  )
}
