import { useMutation } from "@tanstack/react-query";
import { LoaderCircle } from "lucide-react";
import { type FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { login, register } from "../lib/api";
import { useAuth } from "../lib/auth";

type Mode = "signin" | "register";

export function LoginPage() {
  const { status, signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? "/";

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const submit = useMutation({
    mutationFn: () => (mode === "signin" ? login(email, password) : register(email, password)),
    onSuccess: (response) => {
      signIn(response);
      navigate(from, { replace: true });
    },
  });

  if (status === "authenticated") return <Navigate to={from} replace />;

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit.mutate();
  };

  const switchMode = () => {
    setMode(mode === "signin" ? "register" : "signin");
    submit.reset();
  };

  return (
    <div className="flex min-h-dvh items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-sm">
        <p className="mb-2 text-center text-2xl font-semibold tracking-tight text-slate-900">
          Docu<span className="text-indigo-600">Mind</span>
        </p>
        <p className="mb-8 text-center text-sm text-slate-500">
          Ask your documents. Every answer cites its page.
        </p>

        <form
          onSubmit={onSubmit}
          className="space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200"
          data-testid="auth-form"
        >
          <h1 className="text-lg font-semibold text-slate-900">
            {mode === "signin" ? "Sign in" : "Create an account"}
          </h1>

          <label className="block">
            <span className="text-sm font-medium text-slate-700">Email</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none"
              data-testid="email-input"
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium text-slate-700">Password</span>
            <input
              type="password"
              required
              minLength={mode === "register" ? 8 : undefined}
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none"
              data-testid="password-input"
            />
            {mode === "register" && (
              <span className="mt-1 block text-xs text-slate-500">At least 8 characters.</span>
            )}
          </label>

          {submit.isError && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700" role="alert">
              {submit.error.message}
            </p>
          )}

          <button
            type="submit"
            disabled={submit.isPending}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-60"
            data-testid="auth-submit"
          >
            {submit.isPending && <LoaderCircle className="size-4 animate-spin" aria-hidden />}
            {mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-slate-600">
          {mode === "signin" ? "New here?" : "Already have an account?"}{" "}
          <button
            type="button"
            onClick={switchMode}
            className="font-medium text-indigo-600 hover:text-indigo-700"
            data-testid="auth-switch"
          >
            {mode === "signin" ? "Create an account" : "Sign in"}
          </button>
        </p>
      </div>
    </div>
  );
}
