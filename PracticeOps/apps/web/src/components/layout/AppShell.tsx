/**
 * App Shell Component
 *
 * Clean layout with minimal chrome.
 */

import { Outlet } from "react-router-dom";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { DemoBanner } from "@/components/demo/DemoBanner";
import { useAuth } from "@/lib/auth";

export function AppShell() {
  const { isDemoUser } = useAuth();

  return (
    <div className="flex min-h-screen flex-col bg-white">
      <Header />
      {isDemoUser && <DemoBanner />}
      <div className="flex flex-1">
        <Sidebar className="hidden md:block" />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
