/**
 * Auth Layout
 *
 * Clean, minimal layout for auth pages.
 */

import { Link } from "react-router-dom";
import { Logo } from "@/components/brand/Logo";

interface AuthLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export function AuthLayout({ children, title, subtitle }: AuthLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      {/* Header */}
      <header className="h-14 flex items-center px-6 border-b border-neutral-200">
        <Link to="/" className="inline-flex">
          <Logo />
        </Link>
      </header>

      {/* Content */}
      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-8">
            <h1 className="text-xl font-semibold text-neutral-900">{title}</h1>
            {subtitle && (
              <p className="mt-2 text-sm text-neutral-500">{subtitle}</p>
            )}
          </div>
          {children}
        </div>
      </main>
    </div>
  );
}
