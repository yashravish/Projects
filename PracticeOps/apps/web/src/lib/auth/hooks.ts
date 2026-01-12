/**
 * Auth Hooks
 *
 * Custom hooks for accessing authentication state and actions.
 */

import { useContext } from "react";
import { AuthContext, type AuthContextValue } from "./context";

/**
 * Hook to access the auth context.
 *
 * Must be used within an AuthProvider.
 *
 * @returns The auth context value
 * @throws Error if used outside of AuthProvider
 */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (context === null) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}

/**
 * Hook to get the current user.
 *
 * @returns The current user or null if not authenticated
 */
export function useCurrentUser() {
  const { user, isAuthenticated, isLoading } = useAuth();
  return { user, isAuthenticated, isLoading };
}

/**
 * Hook to check if current user is a leader.
 *
 * Leader = SECTION_LEADER or ADMIN role
 *
 * @returns Boolean indicating if user is a leader
 */
export function useIsLeader() {
  const { isLeader, isLoading } = useAuth();
  return { isLeader, isLoading };
}

/**
 * Hook to get the current team context.
 *
 * For MVP, this returns the primary team membership.
 * Future: Will support team switching.
 *
 * @returns The primary team membership or null
 */
export function useTeam() {
  const { primaryTeam, isLoading } = useAuth();
  return { team: primaryTeam, isLoading };
}

