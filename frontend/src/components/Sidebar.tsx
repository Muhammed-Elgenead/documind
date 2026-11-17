import { useQuery } from "@tanstack/react-query";
import { FileText, LogOut, MessageSquare, Plus } from "lucide-react";
import { NavLink, useMatch } from "react-router-dom";

import { fetchHealth, listConversations } from "../lib/api";
import { useAuth } from "../lib/auth";

interface Props {
  onNavigate?: () => void;
}

export function Sidebar({ onNavigate }: Props) {
  // The sidebar lives in the layout route, so read the child route's param via useMatch.
  const conversationId = useMatch("/c/:conversationId")?.params.conversationId;
  const conversations = useQuery({ queryKey: ["conversations"], queryFn: listConversations });
  const health = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
  });

  const { user, signOut } = useAuth();
  const healthy = health.data?.status === "healthy";

  return (
    <nav className="flex h-full flex-col bg-slate-50" aria-label="Main">
      <div className="px-4 pt-5 pb-3">
        <p className="text-lg font-semibold tracking-tight text-slate-900">
          Docu<span className="text-indigo-600">Mind</span>
        </p>
      </div>

      <div className="space-y-1 px-3">
        <NavLink
          to="/"
          end
          onClick={onNavigate}
          className="flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700"
        >
          <Plus className="size-4" aria-hidden /> New chat
        </NavLink>
        <NavLink
          to="/documents"
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${
              isActive ? "bg-white text-slate-900 shadow-sm ring-1 ring-slate-200" : "text-slate-600 hover:bg-white"
            }`
          }
        >
          <FileText className="size-4" aria-hidden /> Documents
        </NavLink>
      </div>

      <p className="mt-6 px-5 pb-1 text-xs font-medium tracking-wide text-slate-400 uppercase">
        Conversations
      </p>
      <ul className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-3" data-testid="conversation-list">
        {conversations.data?.length === 0 && (
          <li className="px-3 py-2 text-sm text-slate-400">No conversations yet</li>
        )}
        {conversations.data?.map((c) => (
          <li key={c.id}>
            <NavLink
              to={`/c/${c.id}`}
              onClick={onNavigate}
              title={c.title}
              className={`flex items-center gap-2 truncate rounded-lg px-3 py-2 text-sm ${
                c.id === conversationId
                  ? "bg-white font-medium text-slate-900 shadow-sm ring-1 ring-slate-200"
                  : "text-slate-600 hover:bg-white"
              }`}
            >
              <MessageSquare className="size-4 shrink-0 text-slate-400" aria-hidden />
              <span className="truncate">{c.title}</span>
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="flex items-center gap-2 border-t border-slate-200 px-5 pt-3 text-sm text-slate-600">
        <span className="min-w-0 flex-1 truncate" title={user?.email} data-testid="user-email">
          {user?.email}
        </span>
        <button
          type="button"
          onClick={signOut}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-slate-500 hover:bg-white hover:text-slate-900"
          data-testid="sign-out"
        >
          <LogOut className="size-3.5" aria-hidden /> Sign out
        </button>
      </div>
      <div className="flex items-center gap-2 px-5 pt-1 pb-3 text-xs text-slate-500">
        <span
          className={`size-2 rounded-full ${
            health.isPending ? "bg-slate-300" : healthy ? "bg-emerald-500" : "bg-red-500"
          }`}
          aria-hidden
        />
        <span data-testid="health-status">
          {health.isPending ? "checking…" : healthy ? "healthy" : "backend unavailable"}
        </span>
      </div>
    </nav>
  );
}
