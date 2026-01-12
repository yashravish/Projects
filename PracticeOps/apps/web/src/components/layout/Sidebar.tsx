/**
 * Sidebar Navigation Component
 *
 * Minimal text-only navigation.
 */

import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";

interface NavItem {
  label: string;
  path: string;
  leaderOnly?: boolean;
  adminOnly?: boolean;
}

const navItems: NavItem[] = [
  { label: "This week", path: "/dashboard" },
  { label: "Tickets", path: "/tickets" },
  { label: "Assignments", path: "/assignments" },
  { label: "Practice logs", path: "/practice" },
  { label: "Section overview", path: "/leader", leaderOnly: true },
  { label: "Team settings", path: "/settings", adminOnly: true },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const { isLeader, primaryTeam } = useAuth();
  const isAdmin = primaryTeam?.role === "ADMIN";

  const filteredItems = navItems.filter((item) => {
    if (item.adminOnly && !isAdmin) return false;
    if (item.leaderOnly && !isLeader) return false;
    return true;
  });

  return (
    <aside className={cn("w-52 flex-shrink-0 border-r border-neutral-200 bg-neutral-50", className)}>
      <nav className="py-6 px-4">
        <ul className="space-y-1">
          {filteredItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    "block px-3 py-2 text-sm rounded transition-colors",
                    isActive
                      ? "text-neutral-900 bg-white shadow-sm"
                      : "text-neutral-600 hover:text-neutral-900 hover:bg-white/50"
                  )
                }
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
