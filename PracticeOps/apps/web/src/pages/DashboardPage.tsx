/**
 * Dashboard Page
 *
 * Task-focused view for preparing for rehearsal.
 * Shows what needs attention, not stats.
 */

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { DashboardSkeleton } from "@/components/ui/loading";
import { NoCycleEmptyState } from "@/components/ui/empty-state";
import { LogPracticeModal } from "@/components/dashboard";
import { getMemberDashboard, ApiClientError } from "@/lib/api/client";
import { useAuth, useTeam } from "@/lib/auth";
import { useNavigate } from "react-router-dom";
import { AlertCircle, RefreshCw, ChevronRight } from "lucide-react";
import { format, parseISO, differenceInDays } from "date-fns";

// Hook to detect mobile viewport
function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return isMobile;
}

export function DashboardPage() {
  const { logout, isLoading: authLoading } = useAuth();
  const { team, isLoading: teamLoading } = useTeam();
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const [logPracticeOpen, setLogPracticeOpen] = useState(false);

  // Fetch dashboard data
  const {
    data: dashboard,
    isLoading: dashboardLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["dashboard", team?.team_id],
    queryFn: () => {
      if (!team?.team_id) throw new Error("No team");
      return getMemberDashboard(team.team_id);
    },
    enabled: !!team?.team_id,
    staleTime: 30 * 1000,
    retry: (failureCount, err) => {
      if (err instanceof ApiClientError && err.code === "UNAUTHORIZED") {
        return false;
      }
      return failureCount < 2;
    },
  });

  // Handle unauthorized error
  useEffect(() => {
    if (error instanceof ApiClientError && error.code === "UNAUTHORIZED") {
      logout();
      navigate("/login", { replace: true });
    }
  }, [error, logout, navigate]);

  const isLoading = authLoading || teamLoading || dashboardLoading || (!!team?.team_id && !dashboard && !error);
  
  if (isLoading) {
    return <DashboardSkeleton />;
  }

  if (error && !(error instanceof ApiClientError && error.code === "UNAUTHORIZED")) {
    return (
      <div className="p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error loading dashboard</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <p>
              {error instanceof ApiClientError
                ? error.message
                : "Failed to load dashboard."}
            </p>
            <Button variant="outline" size="sm" onClick={() => refetch()} className="w-fit">
              <RefreshCw className="mr-2 h-4 w-4" />
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!team) {
    return (
      <div className="p-6">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No team found</AlertTitle>
          <AlertDescription>
            You're not a member of any team yet. Ask your leader for an invite link.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!dashboard?.cycle) {
    return (
      <div className="p-6">
        <NoCycleEmptyState />
      </div>
    );
  }

  const cycleDate = parseISO(dashboard.cycle.date);
  const daysUntil = differenceInDays(cycleDate, new Date());
  const urgentTickets = dashboard.tickets_due_soon.filter(t => t.priority === "BLOCKING");
  const hasWork = dashboard.assignments.length > 0 || dashboard.tickets_due_soon.length > 0;

  return (
    <div className="min-h-full bg-neutral-50">
      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Cycle header */}
        <div className="flex items-start justify-between mb-12">
          <div>
            <div className="text-sm text-neutral-500 mb-1">
              {format(cycleDate, "EEEE, MMMM d")}
            </div>
            <h1 className="text-2xl font-semibold text-neutral-900">
              {daysUntil === 0 ? "Today" : daysUntil === 1 ? "Tomorrow" : `${daysUntil} days`}
            </h1>
            {dashboard.cycle.label && (
              <div className="text-sm text-neutral-500 mt-1">
                {dashboard.cycle.label}
              </div>
            )}
          </div>
          <Button
            onClick={() => setLogPracticeOpen(true)}
            className="bg-neutral-900 hover:bg-neutral-800 text-white"
          >
            Log practice
          </Button>
        </div>

        {/* Work sections */}
        {hasWork ? (
          <div className="space-y-10">
            {/* Urgent tickets */}
            {urgentTickets.length > 0 && (
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <h2 className="text-sm font-medium text-neutral-900">
                    Blocking issues
                  </h2>
                </div>
                <div className="space-y-2">
                  {urgentTickets.map((ticket) => (
                    <Link
                      key={ticket.id}
                      to={`/tickets/${ticket.id}`}
                      className="flex items-start justify-between p-4 bg-red-50 border border-red-100 rounded-lg hover:bg-red-100/50 transition-colors group"
                    >
                      <div>
                        <div className="text-sm font-medium text-neutral-900">
                          {ticket.title}
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-neutral-400 group-hover:text-neutral-600 mt-0.5" />
                    </Link>
                  ))}
                </div>
              </section>
            )}

            {/* Assignments */}
            {dashboard.assignments.length > 0 && (
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-medium text-neutral-900">
                    Assignments
                  </h2>
                  <Link 
                    to="/assignments" 
                    className="text-xs text-neutral-500 hover:text-neutral-700"
                  >
                    View all
                  </Link>
                </div>
                <div className="space-y-1">
                  {dashboard.assignments.slice(0, 5).map((assignment) => (
                    <div
                      key={assignment.id}
                      className="flex items-center justify-between py-3 border-b border-neutral-100 last:border-0"
                    >
                      <div className="flex items-start gap-3">
                        <div 
                          className={`w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0 ${
                            assignment.priority === "BLOCKING" 
                              ? "bg-red-500" 
                              : assignment.priority === "MEDIUM"
                              ? "bg-amber-500"
                              : "bg-neutral-300"
                          }`}
                        />
                        <div>
                          <div className="text-sm text-neutral-900">
                            {assignment.title}
                          </div>
                        </div>
                      </div>
                      {assignment.due_at && (
                        <div className="text-xs text-neutral-400 tabular-nums">
                          {format(parseISO(assignment.due_at), "MMM d")}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Other tickets */}
            {dashboard.tickets_due_soon.filter(t => t.priority !== "BLOCKING").length > 0 && (
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-medium text-neutral-900">
                    Open tickets
                  </h2>
                  <Link 
                    to="/tickets" 
                    className="text-xs text-neutral-500 hover:text-neutral-700"
                  >
                    View all
                  </Link>
                </div>
                <div className="space-y-1">
                  {dashboard.tickets_due_soon
                    .filter(t => t.priority !== "BLOCKING")
                    .slice(0, 5)
                    .map((ticket) => (
                      <Link
                        key={ticket.id}
                        to={`/tickets/${ticket.id}`}
                        className="flex items-center justify-between py-3 border-b border-neutral-100 last:border-0 hover:bg-neutral-50 -mx-2 px-2 rounded transition-colors"
                      >
                        <div className="flex items-start gap-3">
                          <div 
                            className={`w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0 ${
                              ticket.status === "IN_PROGRESS"
                                ? "bg-blue-500"
                                : ticket.status === "BLOCKED"
                                ? "bg-red-500"
                                : "bg-neutral-300"
                            }`}
                          />
                          <div>
                            <div className="text-sm text-neutral-900">
                              {ticket.title}
                            </div>
                          </div>
                        </div>
                        {ticket.due_at && (
                          <div className="text-xs text-neutral-400 tabular-nums">
                            {format(parseISO(ticket.due_at), "MMM d")}
                          </div>
                        )}
                      </Link>
                    ))}
                </div>
              </section>
            )}
          </div>
        ) : (
          /* Empty state */
          <div className="text-center py-16">
            <div className="text-neutral-400 mb-2">Nothing assigned yet</div>
            <p className="text-sm text-neutral-500">
              Log practice to track your preparation
            </p>
          </div>
        )}

        {/* Practice summary footer */}
        {dashboard.weekly_summary.total_sessions > 0 && (
          <div className="mt-16 pt-8 border-t border-neutral-200">
            <div className="flex items-center justify-between text-sm">
              <div className="text-neutral-500">
                <span className="text-neutral-900 font-medium">
                  {dashboard.weekly_summary.total_sessions}
                </span>
                {" "}sessions this week
              </div>
              <Link 
                to="/practice" 
                className="text-neutral-500 hover:text-neutral-700"
              >
                View history
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* Log Practice Modal */}
      <LogPracticeModal
        open={logPracticeOpen}
        onOpenChange={setLogPracticeOpen}
        cycleId={dashboard.cycle?.id || null}
        assignments={dashboard.assignments}
        defaultDuration={dashboard.quick_log_defaults.duration_min_default}
        teamId={team.team_id}
        isMobile={isMobile}
      />
    </div>
  );
}

export { DashboardSkeleton } from "@/components/ui/loading";
