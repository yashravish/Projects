/**
 * Empty State Components
 *
 * Reusable empty state components for consistent UX across the app.
 * Each preset is context-aware and includes appropriate messaging and CTAs.
 *
 * RBAC Safety: CTAs are only rendered when the action is accessible.
 */

import { ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Calendar,
  CheckCircle2,
  ClipboardList,
  Filter,
  Music,
  Search,
  Ticket,
  UserPlus,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick?: () => void;
    href?: string;
  };
  className?: string;
  /** Compact variant for inline empty states */
  compact?: boolean;
}

/**
 * Base EmptyState component - customizable empty state display.
 * Use presets below for common scenarios.
 */
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  compact = false,
}: EmptyStateProps) {
  const content = (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center",
        compact ? "py-6 px-4" : "py-12 px-6"
      )}
    >
      {icon && (
        <div
          className={cn(
            "flex items-center justify-center rounded-full bg-muted mb-4",
            compact ? "h-12 w-12" : "h-16 w-16"
          )}
        >
          <div className={cn(compact ? "h-6 w-6" : "h-8 w-8", "text-muted-foreground")}>
            {icon}
          </div>
        </div>
      )}
      <h3
        className={cn(
          "font-semibold",
          compact ? "text-base" : "text-xl"
        )}
      >
        {title}
      </h3>
      {description && (
        <p
          className={cn(
            "text-muted-foreground mt-2 max-w-sm",
            compact ? "text-sm" : "text-base"
          )}
        >
          {description}
        </p>
      )}
      {action && (
        <div className="mt-4">
          {action.href ? (
            <Button asChild size={compact ? "sm" : "default"}>
              <Link to={action.href}>{action.label}</Link>
            </Button>
          ) : (
            <Button onClick={action.onClick} size={compact ? "sm" : "default"}>
              {action.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );

  if (compact) {
    return <div className={className}>{content}</div>;
  }

  return (
    <Card className={cn("border-dashed", className)}>
      <CardContent className="p-0">{content}</CardContent>
    </Card>
  );
}

// ============================================================================
// PRESETS - Context-specific empty states
// ============================================================================

interface PresetEmptyStateProps {
  className?: string;
  /** Only show CTA if action is accessible to current user */
  showAction?: boolean;
  /** Optional custom action handler */
  onAction?: () => void;
}

/**
 * Empty state for no assignments this cycle
 */
export function NoAssignmentsEmptyState({
  className,
  showAction = false,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<ClipboardList className="h-full w-full" />}
      title="No assignments yet"
      description="Your section leader will add assignments before rehearsal"
      action={
        showAction && onAction
          ? { label: "Create Assignment", onClick: onAction }
          : undefined
      }
      className={className}
    />
  );
}

/**
 * Empty state for no tickets
 */
export function NoTicketsEmptyState({
  className,
  showAction = false,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<CheckCircle2 className="h-full w-full" />}
      title="All clear!"
      description="No tickets to work on right now"
      action={
        showAction && onAction
          ? { label: "Create Ticket", onClick: onAction }
          : undefined
      }
      className={className}
    />
  );
}

/**
 * Empty state for no practice logs
 */
export function NoPracticeLogsEmptyState({
  className,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Music className="h-full w-full" />}
      title="Start your streak!"
      description="Log your first practice session"
      action={onAction ? { label: "Log Practice", onClick: onAction } : undefined}
      className={className}
    />
  );
}

/**
 * Empty state for no team members (leader view)
 */
export function NoTeamMembersEmptyState({
  className,
  showAction = false,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Users className="h-full w-full" />}
      title="Invite your team"
      description="Get your team members on board to start tracking practice"
      action={
        showAction && onAction
          ? { label: "Create Invite", onClick: onAction }
          : undefined
      }
      className={className}
    />
  );
}

/**
 * Empty state for filter returns no results
 */
export function NoFilterResultsEmptyState({
  className,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Filter className="h-full w-full" />}
      title="No matches"
      description="Try adjusting your filters to find what you're looking for"
      action={onAction ? { label: "Clear filters", onClick: onAction } : undefined}
      className={className}
    />
  );
}

/**
 * Empty state for no search results
 */
export function NoSearchResultsEmptyState({
  className,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Search className="h-full w-full" />}
      title="No results found"
      description="Try a different search term"
      action={onAction ? { label: "Clear search", onClick: onAction } : undefined}
      className={className}
    />
  );
}

/**
 * Empty state for no active cycle
 */
export function NoCycleEmptyState({ className }: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Calendar className="h-full w-full" />}
      title="No Rehearsal Scheduled"
      description="There's no active rehearsal cycle right now. Your section leader will set one up when the next rehearsal is scheduled."
      className={className}
    />
  );
}

/**
 * Empty state for no invites
 */
export function NoInvitesEmptyState({
  className,
  showAction = false,
  onAction,
}: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<UserPlus className="h-full w-full" />}
      title="No active invites"
      description="Create invites to add new members to your team"
      action={
        showAction && onAction
          ? { label: "Create Invite", onClick: onAction }
          : undefined
      }
      className={className}
    />
  );
}

/**
 * Empty state for tickets to verify (leader view)
 */
export function NoTicketsToVerifyEmptyState({ className }: PresetEmptyStateProps) {
  return (
    <EmptyState
      icon={<Ticket className="h-full w-full" />}
      title="No tickets pending verification"
      description="Resolved tickets will appear here when ready for review"
      className={className}
      compact
    />
  );
}

