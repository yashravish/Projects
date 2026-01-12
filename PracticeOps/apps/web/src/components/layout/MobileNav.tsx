/**
 * Mobile Navigation Component
 *
 * Sheet-based navigation for mobile devices.
 */

import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Menu,
  LayoutDashboard,
  Ticket,
  FileText,
  Music,
  BarChart3,
  Settings,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";
import { Logo } from "@/components/brand/Logo";

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
  leaderOnly?: boolean;
  adminOnly?: boolean;
}

const navItems: NavItem[] = [
  {
    label: "Dashboard",
    path: "/dashboard",
    icon: <LayoutDashboard className="h-5 w-5" />,
  },
  {
    label: "Tickets",
    path: "/tickets",
    icon: <Ticket className="h-5 w-5" />,
  },
  {
    label: "Assignments",
    path: "/assignments",
    icon: <FileText className="h-5 w-5" />,
  },
  {
    label: "Practice Logs",
    path: "/practice",
    icon: <Music className="h-5 w-5" />,
  },
  {
    label: "Leader Dashboard",
    path: "/leader",
    icon: <BarChart3 className="h-5 w-5" />,
    leaderOnly: true,
  },
  {
    label: "Team Settings",
    path: "/settings",
    icon: <Settings className="h-5 w-5" />,
    adminOnly: true,
  },
];

export function MobileNav() {
  const [open, setOpen] = useState(false);
  const { isLeader, primaryTeam } = useAuth();
  const isAdmin = primaryTeam?.role === "ADMIN";

  const filteredItems = navItems.filter((item) => {
    if (item.adminOnly && !isAdmin) return false;
    if (item.leaderOnly && !isLeader) return false;
    return true;
  });

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72 p-0">
        <SheetHeader className="border-b px-6 py-4">
          <SheetTitle>
            <Logo />
          </SheetTitle>
        </SheetHeader>
        <nav className="flex-1 space-y-1 px-3 py-4">
          {filteredItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>
      </SheetContent>
    </Sheet>
  );
}

