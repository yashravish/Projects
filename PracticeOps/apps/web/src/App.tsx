/**
 * App Component
 *
 * Main application component with route configuration.
 * Implements route guards for auth, guest, and leader routes.
 */

import { Routes, Route, Navigate } from "react-router-dom";

import { RequireAuth, RequireGuest, RequireLeader, RequireAdmin } from "@/components/guards";
import { AppShell } from "@/components/layout";
import {
  LandingPage,
  LoginPage,
  RegisterPage,
  InviteAcceptPage,
  DashboardPage,
  TicketsListPage,
  TicketDetailPage,
  AssignmentsPage,
  PracticeLogsPage,
  LeaderDashboardPage,
  TeamSettingsPage,
} from "@/pages";

export default function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/invites/:token" element={<InviteAcceptPage />} />

      {/* Guest-only routes (redirect to dashboard if authenticated) */}
      <Route
        path="/login"
        element={
          <RequireGuest>
            <LoginPage />
          </RequireGuest>
        }
      />
      <Route
        path="/register"
        element={
          <RequireGuest>
            <RegisterPage />
          </RequireGuest>
        }
      />

      {/* Protected routes (require authentication) */}
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/tickets" element={<TicketsListPage />} />
        <Route path="/tickets/:ticketId" element={<TicketDetailPage />} />
        <Route path="/assignments" element={<AssignmentsPage />} />
        <Route path="/practice" element={<PracticeLogsPage />} />

        {/* Leader-only routes */}
        <Route
          path="/leader"
          element={
            <RequireLeader>
              <LeaderDashboardPage />
            </RequireLeader>
          }
        />

        {/* Admin-only routes */}
        <Route
          path="/settings"
          element={
            <RequireAdmin>
              <TeamSettingsPage />
            </RequireAdmin>
          }
        />
      </Route>

      {/* Landing page - public */}
      <Route path="/" element={<LandingPage />} />

      {/* Catch all - redirect to landing */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
