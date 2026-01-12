/**
 * Header Component
 *
 * Minimal header with team name and user.
 */

import { Link } from "react-router-dom";
import { UserMenu } from "./UserMenu";
import { MobileNav } from "./MobileNav";
import { Logo } from "@/components/brand/Logo";

export function Header() {

  return (
    <header className="sticky top-0 z-50 h-14 flex items-center justify-between border-b border-neutral-200 bg-white px-4 md:px-6">
      {/* Left: Mobile nav + Logo */}
      <div className="flex items-center gap-4">
        <MobileNav />
        <Link to="/dashboard" className="inline-flex">
          <Logo />
        </Link>
      </div>

      {/* Right: User menu */}
      <div className="flex items-center gap-6">
        <UserMenu />
      </div>
    </header>
  );
}
