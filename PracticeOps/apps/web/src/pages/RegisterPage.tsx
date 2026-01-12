/**
 * Register Page
 *
 * Route: /register
 * Access: Guest only (redirects to dashboard if authenticated)
 */

import { AuthLayout } from "@/components/layout/AuthLayout";
import { RegisterForm } from "@/components/auth/RegisterForm";

export function RegisterPage() {
  return (
    <AuthLayout
      title="Create an account"
      subtitle="Get started with PracticeOps"
    >
      <RegisterForm />
    </AuthLayout>
  );
}

