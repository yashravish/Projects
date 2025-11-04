"use client";
import { useState } from "react";
import { signIn, signOut } from "next-auth/react";

export default function SignInOut({ authed }: { authed: boolean }) {
  const [loading, setLoading] = useState(false);

  const handleSignOut = async () => {
    try {
      setLoading(true);
      await signOut();
    } catch (error) {
      console.error("Sign out failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSignIn = async () => {
    try {
      setLoading(true);
      await signIn("google");
    } catch (error) {
      console.error("Sign in failed:", error);
    } finally {
      setLoading(false);
    }
  };

  if (authed) {
    return (
      <button
        onClick={handleSignOut}
        disabled={loading}
        className="px-4 py-1.5 rounded-lg glass border border-slate-500/60 text-slate-100 hover:border-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "…" : "Sign out"}
      </button>
    );
  }

  return (
    <button
      onClick={handleSignIn}
      disabled={loading}
      className="px-4 py-1.5 rounded-lg glass border border-slate-500/60 text-slate-100 hover:border-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {loading ? "…" : "Sign in with Google"}
    </button>
  );
}
