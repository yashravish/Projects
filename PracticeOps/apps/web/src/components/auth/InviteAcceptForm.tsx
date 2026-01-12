/**
 * Invite Accept Form Component
 *
 * Handles invite acceptance with multiple states:
 * - Loading: Fetching invite details
 * - Expired: Invite has expired
 * - Logged In: Show accept button only
 * - Login Required: Email matches existing account
 * - New Account: Show registration form
 */

import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  Loader2,
  Eye,
  EyeOff,
  CheckCircle,
  Users,
  Shield,
  User,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/lib/auth";
import { api, setTokens, ApiClientError } from "@/lib/api/client";
import type { Role } from "@/lib/api/types";

// Role display helpers
const roleConfig: Record<Role, { label: string; icon: React.ReactNode }> = {
  MEMBER: { label: "Member", icon: <User className="h-3 w-3" /> },
  SECTION_LEADER: {
    label: "Section Leader",
    icon: <Shield className="h-3 w-3" />,
  },
  ADMIN: { label: "Admin", icon: <Shield className="h-3 w-3" /> },
};

export function InviteAcceptForm() {
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();
  const { isAuthenticated, refreshUser } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [needsLogin, setNeedsLogin] = useState(false);

  // Fetch invite preview
  const {
    data: invite,
    isLoading,
    error: fetchError,
  } = useQuery({
    queryKey: ["invite", token],
    queryFn: () => api.getInvitePreview(token!),
    enabled: !!token,
    retry: false,
  });

  // Pre-fill email from invite
  useEffect(() => {
    if (invite?.email) {
      setEmail(invite.email);
    }
  }, [invite?.email]);

  const handleAccept = async () => {
    if (!token) return;
    setError(null);
    setIsSubmitting(true);

    try {
      if (isAuthenticated) {
        // Logged in: just accept
        await api.acceptInvite(token, {});
        await refreshUser();
        navigate("/dashboard", { replace: true });
      } else {
        // Not logged in: need to create account
        if (!name.trim()) {
          setError("Please enter your name");
          setIsSubmitting(false);
          return;
        }
        if (!email.trim()) {
          setError("Please enter your email");
          setIsSubmitting(false);
          return;
        }
        if (password.length < 8) {
          setError("Password must be at least 8 characters");
          setIsSubmitting(false);
          return;
        }

        const response = await api.acceptInvite(token, {
          name,
          email,
          password,
        });

        if (response.access_token && response.refresh_token) {
          setTokens(response.access_token, response.refresh_token);
          await refreshUser();
        }

        navigate("/dashboard", { replace: true });
      }
    } catch (err) {
      if (err instanceof ApiClientError) {
        if (
          err.code === "CONFLICT" &&
          err.message.includes("Please log in")
        ) {
          // Account exists, need to login first
          setNeedsLogin(true);
          setError(null);
        } else if (err.code === "FORBIDDEN") {
          setError("This invite has expired");
        } else if (err.code === "CONFLICT") {
          setError(err.message);
        } else {
          setError(err.message);
        }
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
      setIsSubmitting(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="space-y-3">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-40" />
        </div>
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  // Error fetching invite
  if (fetchError) {
    const isNotFound =
      fetchError instanceof ApiClientError &&
      fetchError.code === "NOT_FOUND";

    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {isNotFound
            ? "This invite link is invalid or has already been used."
            : "Unable to load invite details. Please try again."}
        </AlertDescription>
      </Alert>
    );
  }

  // No invite found
  if (!invite) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>Invite not found</AlertDescription>
      </Alert>
    );
  }

  // Expired invite
  if (invite.expired) {
    return (
      <div className="space-y-4">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            This invite has expired or has already been used.
          </AlertDescription>
        </Alert>
        <p className="text-center text-sm text-muted-foreground">
          Please contact your team admin for a new invite.
        </p>
      </div>
    );
  }

  // Need to login first
  if (needsLogin) {
    return (
      <div className="space-y-4">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            An account with this email already exists. Please log in to accept
            this invite.
          </AlertDescription>
        </Alert>

        <InviteDetails invite={invite} />

        <Button asChild className="w-full">
          <Link to={`/login?redirect=/invites/${token}`}>
            Login to Accept Invite
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <InviteDetails invite={invite} />

      <Separator />

      {isAuthenticated ? (
        // Logged in: just show accept button
        <div className="space-y-4">
          <div className="rounded-lg bg-muted/50 p-4 text-center">
            <CheckCircle className="mx-auto mb-2 h-8 w-8 text-green-500" />
            <p className="text-sm text-muted-foreground">
              You're logged in and ready to join!
            </p>
          </div>
          <Button
            onClick={handleAccept}
            className="w-full"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Accepting...
              </>
            ) : (
              "Accept Invite"
            )}
          </Button>
        </div>
      ) : (
        // Not logged in: show registration form
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAccept();
          }}
          className="space-y-4"
        >
          <p className="text-center text-sm text-muted-foreground">
            Create an account to join the team
          </p>

          <div className="space-y-2">
            <Label htmlFor="name">Display Name</Label>
            <Input
              id="name"
              type="text"
              placeholder="Your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              autoComplete="name"
              autoFocus
              maxLength={100}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              disabled={!!invite.email}
            />
            {invite.email && (
              <p className="text-xs text-muted-foreground">
                Email is set by the invite
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? "text" : "password"}
                placeholder="Min. 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="new-password"
                minLength={8}
                maxLength={128}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                tabIndex={-1}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <Button type="submit" className="w-full" disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating account...
              </>
            ) : (
              "Create Account & Join Team"
            )}
          </Button>

          <p className="text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link
              to={`/login?redirect=/invites/${token}`}
              className="font-medium text-primary hover:underline"
            >
              Sign in
            </Link>
          </p>
        </form>
      )}
    </div>
  );
}

// Invite details component
function InviteDetails({
  invite,
}: {
  invite: {
    team_name: string;
    role: Role;
    section: string | null;
  };
}) {
  const { label, icon } = roleConfig[invite.role];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
          <Users className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold">{invite.team_name}</h3>
          <p className="text-sm text-muted-foreground">You've been invited!</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary" className="gap-1">
          {icon}
          {label}
        </Badge>
        {invite.section && (
          <Badge variant="outline">{invite.section} Section</Badge>
        )}
      </div>
    </div>
  );
}

