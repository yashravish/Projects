/**
 * User Menu Component
 *
 * Dropdown menu with user info, role badge, and actions.
 */

import { useNavigate } from "react-router-dom";
import {
  LogOut,
  User,
  Shield,
  Crown,
  ChevronDown,
} from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/lib/auth";
import type { Role } from "@/lib/api/types";

// Role display config
const roleConfig: Record<
  Role,
  { label: string; icon: React.ReactNode; variant: "default" | "secondary" | "outline" }
> = {
  MEMBER: {
    label: "Member",
    icon: <User className="h-3 w-3" />,
    variant: "outline",
  },
  SECTION_LEADER: {
    label: "Section Leader",
    icon: <Shield className="h-3 w-3" />,
    variant: "secondary",
  },
  ADMIN: {
    label: "Admin",
    icon: <Crown className="h-3 w-3" />,
    variant: "default",
  },
};

export function UserMenu() {
  const navigate = useNavigate();
  const { user, primaryTeam, logout } = useAuth();

  if (!user) return null;

  const initials = user.name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const role = primaryTeam?.role;
  const roleInfo = role ? roleConfig[role] : null;

  const handleLogout = () => {
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="h-auto gap-2 px-2 py-1.5"
          aria-label="User menu"
        >
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-primary text-primary-foreground text-xs">
              {initials}
            </AvatarFallback>
          </Avatar>
          <div className="hidden items-center gap-2 md:flex">
            <span className="text-sm font-medium leading-none">{user.name}</span>
            {roleInfo && (
              <Badge variant="secondary" className="text-[10px] font-normal px-1.5 py-0 h-4 bg-neutral-100 text-neutral-600 border-0">
                {roleInfo.label}
              </Badge>
            )}
          </div>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium">{user.name}</p>
            <p className="text-xs text-muted-foreground">{user.email}</p>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem disabled>
          <User className="mr-2 h-4 w-4" />
          Profile
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={handleLogout}
          className="text-destructive focus:text-destructive"
        >
          <LogOut className="mr-2 h-4 w-4" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

