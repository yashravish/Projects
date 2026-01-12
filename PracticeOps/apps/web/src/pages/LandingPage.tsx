/**
 * Landing Page
 *
 * Direct entry point. No marketing fluff.
 * Shows what the product does through interface preview.
 */

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Logo } from "@/components/brand/Logo";
import { useAuth } from "@/lib/auth";

export function LandingPage() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [isDemoLoading, setIsDemoLoading] = useState(false);

  const handleDemoLogin = async () => {
    setIsDemoLoading(true);
    try {
      await login({
        email: import.meta.env.VITE_DEMO_EMAIL || "olivia@practiceops.app",
        password: import.meta.env.VITE_DEMO_PASSWORD || "demo1234",
      });
      navigate("/dashboard");
    } catch (error) {
      console.error("Demo login failed:", error);
      // Silently fail - user can try regular login if demo fails
    } finally {
      setIsDemoLoading(false);
    }
  };
  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Minimal header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-neutral-50/80 backdrop-blur-sm border-b border-neutral-200 px-6">
        <div className="max-w-screen-xl mx-auto h-14 flex items-center justify-between">
          <Link to="/" className="inline-flex">
            <Logo />
          </Link>
          <Link
            to="/login"
            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
          >
            Sign in
          </Link>
        </div>
      </header>

      {/* Main content */}
      <main className="pt-14">
        {/* Entry section */}
        <section className="px-6 pt-24 pb-20">
          <div className="max-w-screen-xl mx-auto">
            <div className="max-w-2xl">
              <h1 className="font-serif text-4xl md:text-5xl text-neutral-900 leading-tight tracking-tight">
                Know where your ensemble stands before you walk into rehearsal.
              </h1>
              <p className="mt-6 text-lg text-neutral-600 leading-relaxed max-w-xl">
                Members log practice. Section leaders track compliance. 
                Problems surface as tickets that get claimed and resolved.
              </p>
              <div className="mt-10 flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <button
                  onClick={handleDemoLogin}
                  disabled={isDemoLoading}
                  className="inline-flex items-center justify-center h-11 px-6 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDemoLoading ? "Loading..." : "Try the Demo"}
                </button>
                <Link
                  to="/register"
                  className="inline-flex items-center justify-center h-11 px-6 border border-neutral-300 text-neutral-900 text-sm font-medium rounded-md hover:bg-neutral-100 transition-colors"
                >
                  Create a team
                </Link>
              </div>
              <p className="mt-4 text-sm text-neutral-500">
                Demo workspace · Data may reset periodically
              </p>
            </div>
          </div>
        </section>

        {/* Interface preview - shows the actual product */}
        <section className="px-6 pb-32">
          <div className="max-w-screen-xl mx-auto">
            <div className="bg-white border border-neutral-200 rounded-lg shadow-sm overflow-hidden">
              {/* Mock browser chrome */}
              <div className="h-10 bg-neutral-100 border-b border-neutral-200 flex items-center px-4 gap-2">
                <div className="w-3 h-3 rounded-full bg-neutral-300" />
                <div className="w-3 h-3 rounded-full bg-neutral-300" />
                <div className="w-3 h-3 rounded-full bg-neutral-300" />
              </div>
              
              {/* Product preview */}
              <div className="grid grid-cols-[240px,1fr] min-h-[480px]">
                {/* Sidebar preview */}
                <div className="border-r border-neutral-200 p-4">
                  <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-4">
                    Soprano Section
                  </div>
                  <nav className="space-y-1">
                    <div className="px-3 py-2 text-sm text-neutral-900 bg-neutral-100 rounded">
                      This week
                    </div>
                    <div className="px-3 py-2 text-sm text-neutral-500">
                      Practice logs
                    </div>
                    <div className="px-3 py-2 text-sm text-neutral-500">
                      Open tickets
                    </div>
                  </nav>
                  
                  <div className="mt-8 pt-4 border-t border-neutral-100">
                    <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-3">
                      Rehearsal
                    </div>
                    <div className="text-2xl font-semibold text-neutral-900 tabular-nums">
                      3 days
                    </div>
                    <div className="text-sm text-neutral-500 mt-1">
                      Sunday, Jan 5
                    </div>
                  </div>
                </div>

                {/* Main content preview */}
                <div className="p-6">
                  <div className="flex items-start justify-between mb-8">
                    <div>
                      <h2 className="text-lg font-medium text-neutral-900">
                        Section compliance
                      </h2>
                      <p className="text-sm text-neutral-500 mt-1">
                        8 of 12 members logged this week
                      </p>
                    </div>
                    <div className="text-3xl font-semibold text-neutral-900 tabular-nums">
                      67%
                    </div>
                  </div>

                  {/* Member list preview */}
                  <div className="space-y-px bg-neutral-100 rounded-md overflow-hidden">
                    {[
                      { name: "Sarah M.", days: 4, status: "on-track" },
                      { name: "Marcus C.", days: 3, status: "on-track" },
                      { name: "Elena R.", days: 2, status: "behind" },
                      { name: "James T.", days: 0, status: "none" },
                    ].map((member, i) => (
                      <div 
                        key={i}
                        className="flex items-center justify-between px-4 py-3 bg-white"
                      >
                        <span className="text-sm text-neutral-900">{member.name}</span>
                        <div className="flex items-center gap-3">
                          <span className="text-sm text-neutral-500 tabular-nums">
                            {member.days} days
                          </span>
                          <span 
                            className={`w-2 h-2 rounded-full ${
                              member.status === "on-track" 
                                ? "bg-emerald-500" 
                                : member.status === "behind"
                                ? "bg-amber-500"
                                : "bg-neutral-300"
                            }`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Tickets preview */}
                  <div className="mt-8">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-medium text-neutral-900">
                        Open tickets
                      </h3>
                      <span className="text-xs text-neutral-400">2 blocking</span>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-start gap-3 p-3 bg-red-50 border border-red-100 rounded">
                        <span className="w-1.5 h-1.5 mt-2 rounded-full bg-red-500 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-neutral-900">
                            Pitch drift in mm. 24-32
                          </div>
                          <div className="text-xs text-neutral-500 mt-1">
                            Ave Maria · Blocking
                          </div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3 p-3 bg-neutral-50 border border-neutral-200 rounded">
                        <span className="w-1.5 h-1.5 mt-2 rounded-full bg-neutral-400 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-neutral-900">
                            Memory gaps in verse 2
                          </div>
                          <div className="text-xs text-neutral-500 mt-1">
                            Hallelujah · Normal
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How it works - minimal, no icons */}
        <section className="px-6 py-20 bg-white border-t border-neutral-200">
          <div className="max-w-screen-xl mx-auto">
            <div className="grid md:grid-cols-3 gap-12 md:gap-8">
              <div>
                <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-3">
                  Members
                </div>
                <p className="text-neutral-600 leading-relaxed">
                  Log practice sessions in under a minute. 
                  Note duration, rate quality, flag blockers. 
                  Create tickets for problems that need attention.
                </p>
              </div>
              <div>
                <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-3">
                  Section leaders
                </div>
                <p className="text-neutral-600 leading-relaxed">
                  See who's practicing and who's not. 
                  Review tickets from your section.
                  Verify resolved issues before rehearsal.
                </p>
              </div>
              <div>
                <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-3">
                  Admins
                </div>
                <p className="text-neutral-600 leading-relaxed">
                  Full visibility across all sections. 
                  Schedule rehearsal cycles.
                  Manage team membership and roles.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="px-6 py-8 border-t border-neutral-200">
        <div className="max-w-screen-xl mx-auto flex items-center justify-between">
          <Logo />
          <Link 
            to="/register" 
            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
          >
            Get started
          </Link>
        </div>
      </footer>
    </div>
  );
}
