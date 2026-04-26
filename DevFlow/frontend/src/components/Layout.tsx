import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  Beaker,
  BookOpen,
  Bot,
  Bug,
  Flag,
  LayoutDashboard,
  ListChecks,
  Package,
  Server,
} from "lucide-react";
import { clsx } from "clsx";

const nav = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/projects", label: "Projects", icon: Package },
  { to: "/pipelines", label: "Pipeline Runs", icon: ListChecks },
  { to: "/deployments", label: "Deployments", icon: Server },
  { to: "/flags", label: "Feature Flags", icon: Flag },
  { to: "/experiments", label: "A/B Experiments", icon: Beaker },
  { to: "/defects", label: "Defects", icon: Bug },
  { to: "/knowledge", label: "Knowledge Base", icon: BookOpen },
  { to: "/ai", label: "AI Failure Analyzer", icon: Bot },
  { to: "/metrics", label: "Metrics", icon: Activity },
];

export function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 flex-shrink-0 border-r border-slate-800/80 bg-slate-900/30 p-4 md:block">
        <div className="mb-6 flex items-center gap-2 rounded-2xl border border-slate-800/60 bg-slate-900/60 p-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-brand-500 to-indigo-500 shadow" />
          <div>
            <p className="text-xs uppercase tracking-widest text-slate-400">DevFlow</p>
            <p className="text-sm font-semibold text-white">Release Intelligence</p>
          </div>
        </div>
        <nav className="space-y-1 text-sm">
          {nav.map((item) => {
            const I = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    "group flex items-center gap-2 rounded-xl px-3 py-2 transition",
                    isActive
                      ? "bg-slate-800/80 text-white shadow-inner"
                      : "text-slate-300 hover:bg-slate-800/50",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <I className={clsx("h-4 w-4", isActive ? "text-brand-200" : "text-slate-400")} />
                    <span className="font-medium">{item.label}</span>
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>
        <p className="mt-6 text-xs text-slate-500">
          API docs:{" "}
          <a className="text-brand-200 underline" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
            /docs
          </a>
        </p>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-slate-900/50 px-4 py-3 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between">
            <p className="text-sm text-slate-400">Developer tooling · CI/CD simulation · Observability</p>
            <a className="df-btn text-xs" href="https://github.com" target="_blank" rel="noreferrer">
              View source
            </a>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
