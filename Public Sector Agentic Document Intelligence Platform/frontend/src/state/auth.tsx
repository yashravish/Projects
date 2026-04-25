import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { tokenStore } from '@/api/client';
import { fetchMe, login as apiLogin, register as apiRegister } from '@/api/auth';
import type { User } from '@/api/schemas';

interface AuthState {
  user: User | null;
  status: 'idle' | 'loading' | 'authenticated' | 'unauthenticated';
  error: string | null;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, orgName: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(() => ({
    user: null,
    status: tokenStore.getAccess() ? 'loading' : 'unauthenticated',
    error: null,
  }));

  const hydrate = useCallback(async () => {
    if (!tokenStore.getAccess()) {
      setState({ user: null, status: 'unauthenticated', error: null });
      return;
    }
    try {
      const user = await fetchMe();
      setState({ user, status: 'authenticated', error: null });
    } catch {
      tokenStore.clear();
      setState({ user: null, status: 'unauthenticated', error: null });
    }
  }, []);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  const login = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, status: 'loading', error: null }));
    try {
      const tokens = await apiLogin(email, password);
      tokenStore.set(tokens.access_token, tokens.refresh_token);
      const user = await fetchMe();
      setState({ user, status: 'authenticated', error: null });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'login failed';
      setState({ user: null, status: 'unauthenticated', error: message });
      throw err;
    }
  }, []);

  const register = useCallback(
    async (email: string, password: string, orgName: string) => {
      setState((s) => ({ ...s, status: 'loading', error: null }));
      try {
        const resp = await apiRegister(email, password, orgName);
        tokenStore.set(resp.access_token, resp.refresh_token);
        setState({ user: resp.user, status: 'authenticated', error: null });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'registration failed';
        setState({ user: null, status: 'unauthenticated', error: message });
        throw err;
      }
    },
    [],
  );

  const logout = useCallback(() => {
    tokenStore.clear();
    setState({ user: null, status: 'unauthenticated', error: null });
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ ...state, login, register, logout }),
    [state, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within <AuthProvider>');
  }
  return ctx;
}
