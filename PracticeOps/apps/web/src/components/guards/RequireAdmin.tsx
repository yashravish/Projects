/**
 * RequireAdmin Guard
 *
 * Protects routes that require ADMIN role only.
 * Redirects to /dashboard if not an admin.
 */

import { Navigate } from "react-router-dom";
import { useAuth } from "@/lib/auth";
import { LoadingScreen } from "@/components/layout/LoadingScreen";

interface RequireAdminProps {
  children: React.ReactNode;
}

export function RequireAdmin({ children }: RequireAdminProps) {
  const { isAuthenticated, primaryTeam, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (primaryTeam?.role !== "ADMIN") {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

