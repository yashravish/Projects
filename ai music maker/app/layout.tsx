import type React from "react"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Song Creator",
  description: "Create songs with AI"
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
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
