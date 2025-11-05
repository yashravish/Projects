"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Music, Plus, Home } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Navigation() {
  const pathname = usePathname()

  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/10">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="p-2 rounded-xl bg-gradient-to-br from-red-600 to-red-500 shadow-lg transition-transform group-hover:scale-105">
              <Music className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-semibold text-slate-100 tracking-tight">
              MusicAI Studio
            </span>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-3">
            <Link
              href="/"
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                pathname === "/"
                  ? "bg-white/10 text-slate-100"
                  : "text-slate-400 hover:text-slate-100 hover:bg-white/5"
              }`}
            >
              <Home className="h-4 w-4 inline mr-2" />
              Dashboard
            </Link>

            <Link href="/songs/new">
              <Button className="bg-red-600 hover:bg-red-500 text-white rounded-xl px-4 py-2 text-sm font-medium transition-all hover:glow shadow-lg">
                <Plus className="h-4 w-4 mr-2" />
                New Song
              </Button>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
