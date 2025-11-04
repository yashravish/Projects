import "@/styles/globals.css";
import type { ReactNode } from "react";
import Link from "next/link";
import SignInOut from "@/components/SignInOut";
import { Providers } from "./providers";
import { auth } from "@/lib/auth-helpers";

export const metadata = {
  title: "LLM Playground",
  description: "OpenAI-only LLM playground with streaming"
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  let session = null;

  try {
    session = await auth();
  } catch (error) {
    console.error("Auth error in layout:", error);
    // Continue rendering without session
  }

  return (
    <html lang="en">
      <body className="min-h-screen text-slate-100">
        <Providers>
          <header className="py-6">
            <div className="max-w-4xl mx-auto px-5 py-3 glass-strong rounded-2xl border shadow-glass flex items-center justify-between">
              <div className="flex items-center gap-6">
                <h1 className="font-semibold tracking-tight">LLM Playground</h1>
                <nav className="flex gap-3">
                  <Link
                    href="/"
                    className="text-sm text-slate-300 hover:text-white transition-colors px-2 py-1 rounded hover:bg-slate-700/30"
                  >
                    Playground
                  </Link>
                  <Link
                    href="/web"
                    className="text-sm text-slate-300 hover:text-white transition-colors px-2 py-1 rounded hover:bg-slate-700/30"
                  >
                    Web
                  </Link>
                  <Link
                    href="/support"
                    className="text-sm text-slate-300 hover:text-white transition-colors px-2 py-1 rounded hover:bg-slate-700/30"
                  >
                    Support
                  </Link>
                </nav>
              </div>
              <SignInOut authed={Boolean(session)} />
            </div>
          </header>
          {children}
        </Providers>
      </body>
    </html>
  );
}
