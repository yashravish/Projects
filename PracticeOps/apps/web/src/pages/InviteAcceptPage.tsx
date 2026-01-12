/**
 * Invite Accept Page
 *
 * Route: /invites/:token
 * Access: Public (no auth required)
 *
 * Displays invite details and allows acceptance.
 * Handles multiple states: loading, expired, logged in, needs login, new account.
 */

import { AuthLayout } from "@/components/layout/AuthLayout";
import { InviteAcceptForm } from "@/components/auth/InviteAcceptForm";

export function InviteAcceptPage() {
  return (
    <AuthLayout
      title="Join a Team"
      subtitle="You've been invited to join a practice team"
    >
      <InviteAcceptForm />
    </AuthLayout>
  );
}

