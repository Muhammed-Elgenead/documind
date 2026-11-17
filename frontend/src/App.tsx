import { LoaderCircle } from "lucide-react";
import { Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";

import { Layout } from "./components/Layout";
import { useAuth } from "./lib/auth";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { LoginPage } from "./pages/LoginPage";

function RequireAuth() {
  const { status } = useAuth();
  const location = useLocation();
  if (status === "loading") {
    return (
      <div className="flex h-dvh items-center justify-center text-slate-400">
        <LoaderCircle className="size-6 animate-spin" aria-label="Loading" />
      </div>
    );
  }
  if (status === "anonymous") {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return <Outlet />;
}

export default function App() {
  return (
    <Routes>
      <Route path="login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<Layout />}>
          <Route index element={<ChatPage />} />
          <Route path="c/:conversationId" element={<ChatPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Route>
    </Routes>
  );
}
