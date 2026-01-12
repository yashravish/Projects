/**
 * Auth Context Provider
 *
 * Manages authentication state across the application.
 * Provides user info, team membership, and auth actions.
 */

import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  api,
  clearTokens,
  getAccessToken,
  setTokens,
  ApiClientError,
} from "@/lib/api/client";
import type {
  User,
  TeamMembership,
  LoginRequest,
  RegisterRequest,
} from "@/lib/api/types";

// =============================================================================
// Types
// =============================================================================

export interface AuthState {
  user: User | null;
  primaryTeam: TeamMembership | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface AuthContextValue extends AuthState {
  login: (data: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  isLeader: boolean;
  isDemoUser: boolean;
}

// =============================================================================
// Context
// =============================================================================

export const AuthContext = createContext<AuthContextValue | null>(null);

// =============================================================================
// Provider
// =============================================================================

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const queryClient = useQueryClient();

  const [state, setState] = useState<AuthState>({
    user: null,
    primaryTeam: null,
    isAuthenticated: false,
    isLoading: true,
    error: null,
  });

  // Check if user is a leader (SECTION_LEADER or ADMIN)
  const isLeader = useMemo(() => {
    if (!state.primaryTeam) return false;
    return (
      state.primaryTeam.role === "SECTION_LEADER" ||
      state.primaryTeam.role === "ADMIN"
    );
  }, [state.primaryTeam]);

  // Check if user is a demo account
  const isDemoUser = useMemo(() => {
    if (!state.user) return false;
    return state.user.email.endsWith("@practiceops.app");
  }, [state.user]);

  // Fetch current user info
  const refreshUser = useCallback(async () => {
    const token = getAccessToken();
    if (!token) {
      setState({
        user: null,
        primaryTeam: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
      return;
    }

    try {
      const response = await api.getMe();
      setState({
        user: response.user,
        primaryTeam: response.primary_team,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      // Token is invalid or expired
      clearTokens();
      setState({
        user: null,
        primaryTeam: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    }
  }, []);

  // Login
  const login = useCallback(
    async (data: LoginRequest) => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const response = await api.login(data);
        setTokens(response.access_token, response.refresh_token);

        // Fetch full user info including team membership
        await refreshUser();
      } catch (error) {
        const message =
          error instanceof ApiClientError
            ? error.message
            : "An unexpected error occurred";
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: message,
        }));
        throw error;
      }
    },
    [refreshUser]
  );

  // Register
  const register = useCallback(
    async (data: RegisterRequest) => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const response = await api.register(data);
        setTokens(response.access_token, response.refresh_token);

        // Set initial state (new users don't have a team yet)
        setState({
          user: response.user,
          primaryTeam: null,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
      } catch (error) {
        const message =
          error instanceof ApiClientError
            ? error.message
            : "An unexpected error occurred";
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: message,
        }));
        throw error;
      }
    },
    []
  );

  // Logout
  const logout = useCallback(() => {
    clearTokens();
    queryClient.clear();
    setState({
      user: null,
      primaryTeam: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });
  }, [queryClient]);

  // Initialize auth state on mount
  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const value = useMemo<AuthContextValue>(
    () => ({
      ...state,
      login,
      register,
      logout,
      refreshUser,
      isLeader,
      isDemoUser,
    }),
    [state, login, register, logout, refreshUser, isLeader, isDemoUser]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

