import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { type AuthResponse, fetchMe, getToken, setToken, UNAUTHORIZED_EVENT, type User } from "./api";

type Status = "loading" | "authenticated" | "anonymous";

interface AuthState {
  status: Status;
  user: User | null;
  signIn: (response: AuthResponse) => void;
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [token, setTokenState] = useState(getToken);

  const me = useQuery({
    queryKey: ["me", token],
    queryFn: fetchMe,
    enabled: token !== null,
    retry: false,
    staleTime: Infinity,
  });

  const signOut = useCallback(() => {
    setToken(null);
    setTokenState(null);
    // Drop every cached query so the next user never sees this user's data.
    queryClient.clear();
  }, [queryClient]);

  const signIn = useCallback(
    (response: AuthResponse) => {
      queryClient.clear();
      setToken(response.access_token);
      queryClient.setQueryData(["me", response.access_token], response.user);
      setTokenState(response.access_token);
    },
    [queryClient],
  );

  // Any request that comes back 401 (expired or revoked token) ends the session.
  useEffect(() => {
    const onUnauthorized = () => signOut();
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, [signOut]);

  const value = useMemo<AuthState>(() => {
    let status: Status = "anonymous";
    if (token !== null) status = me.data ? "authenticated" : me.isPending ? "loading" : "anonymous";
    return { status, user: me.data ?? null, signIn, signOut };
  }, [token, me.data, me.isPending, signIn, signOut]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
