/**
 * RequireLeader Guard
 *
 * Protects routes that require leader role (SECTION_LEADER or ADMIN).
 * Redirects to /dashboard if not a leader.
 */

import { Navigate } from "react-router-dom";
import { useAuth } from "@/lib/auth";
import { LoadingScreen } from "@/components/layout/LoadingScreen";

interface RequireLeaderProps {
  children: React.ReactNode;
}

export function RequireLeader({ children }: RequireLeaderProps) {
  const { isAuthenticated, isLeader, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (!isLeader) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

