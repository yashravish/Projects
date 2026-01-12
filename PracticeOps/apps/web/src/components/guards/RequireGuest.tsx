/**
 * RequireGuest Guard
 *
 * Protects auth pages (login, register) from authenticated users.
 * Redirects to /dashboard if already authenticated.
 */

import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/lib/auth";
import { LoadingScreen } from "@/components/layout/LoadingScreen";

interface RequireGuestProps {
  children: React.ReactNode;
}

export function RequireGuest({ children }: RequireGuestProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (isAuthenticated) {
    // Check if there's a redirect URL from a protected route
    const from = (location.state as { from?: Location })?.from?.pathname;
    return <Navigate to={from || "/dashboard"} replace />;
  }

  return <>{children}</>;
}

