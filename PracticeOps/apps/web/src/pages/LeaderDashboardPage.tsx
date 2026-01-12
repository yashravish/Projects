/**
 * Leader Dashboard Page
 *
 * Route: /leader
 * Access: SECTION_LEADER or ADMIN only
 *
 * Displays team health metrics, compliance data, risk summary,
 * and allows verification of resolved tickets.
 *
 * Privacy: Private ticket aggregates show counts only - NO identifiers.
 */

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ButtonSpinner, TableSkeleton, CardSkeleton } from "@/components/ui/loading";
import { NoTicketsToVerifyEmptyState } from "@/components/ui/empty-state";
import { ComplianceInsightsCard } from "@/components/leader/ComplianceInsightsCard";
import { toastSuccess, toastError } from "@/lib/toast";
import {
  Users,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Shield,
  ChevronUp,
  ChevronDown,
  AlertCircle,
  Lock,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { getLeaderDashboard, getComplianceInsights, verifyTicket } from "@/lib/api/client";
import type { TicketVisible } from "@/lib/api/types";

// =============================================================================
// Helper Components
// =============================================================================

function ComplianceRadial({ percentage }: { percentage: number }) {
  const displayPct = Math.round(percentage * 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage * circumference);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg className="w-32 h-32 -rotate-90" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-muted/20"
        />
        {/* Progress circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className={displayPct >= 70 ? "text-emerald-500" : displayPct >= 40 ? "text-amber-500" : "text-rose-500"}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-3xl font-bold">{displayPct}%</span>
        <span className="text-xs text-muted-foreground">compliance</span>
      </div>
    </div>
  );
}

function RiskCard({
  title,
  count,
  icon: Icon,
  variant = "default",
  onClick,
}: {
  title: string;
  count: number;
  icon: typeof AlertTriangle;
  variant?: "default" | "warning" | "danger";
  onClick?: () => void;
}) {
  const colorClass = variant === "danger"
    ? "border-rose-500/50 bg-rose-500/5"
    : variant === "warning"
      ? "border-amber-500/50 bg-amber-500/5"
      : "";
  const iconClass = variant === "danger"
    ? "text-rose-500"
    : variant === "warning"
      ? "text-amber-500"
      : "text-muted-foreground";

  return (
    <Card
      className={`${colorClass} ${onClick ? "cursor-pointer hover:shadow-md transition-shadow" : ""}`}
      onClick={onClick}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1 px-4 pt-4">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 ${iconClass}`} />
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-1">
        <div className="text-2xl font-bold">{count}</div>
        {count >= 3 && variant === "danger" && (
          <Badge variant="destructive" className="mt-1 text-xs">
            Action Required
          </Badge>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function LeaderDashboardPage() {
  const { primaryTeam } = useAuth();
  const queryClient = useQueryClient();

  const [sortField, setSortField] = useState<"name" | "section" | "days_logged_7d">("days_logged_7d");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [selectedTicket, setSelectedTicket] = useState<TicketVisible | null>(null);
  const [verifyNote, setVerifyNote] = useState("");

  // Fetch leader dashboard data
  const {
    data: dashboard,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["leader-dashboard", primaryTeam?.team_id],
    queryFn: () => getLeaderDashboard(primaryTeam!.team_id),
    enabled: !!primaryTeam?.team_id,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const {
    data: complianceInsights,
    isLoading: isInsightsLoading,
    error: insightsError,
  } = useQuery({
    queryKey: ["leader-compliance-insights", primaryTeam?.team_id],
    queryFn: () => getComplianceInsights(primaryTeam!.team_id),
    enabled: !!primaryTeam?.team_id,
    refetchInterval: 30000,
  });

  // Verify ticket mutation
  const verifyMutation = useMutation({
    mutationFn: ({ ticketId, content }: { ticketId: string; content?: string }) =>
      verifyTicket(ticketId, { content }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["leader-dashboard"] });
      setSelectedTicket(null);
      setVerifyNote("");
      toastSuccess("Ticket verified successfully");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to verify ticket");
    },
  });

  // Sort members by selected field
  const sortedMembers = useMemo(() => {
    if (!dashboard?.compliance.practice_days_by_member) return [];

    return [...dashboard.compliance.practice_days_by_member].sort((a, b) => {
      let comparison = 0;
      if (sortField === "name") {
        comparison = a.name.localeCompare(b.name);
      } else if (sortField === "section") {
        comparison = (a.section || "").localeCompare(b.section || "");
      } else {
        comparison = a.days_logged_7d - b.days_logged_7d;
      }
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }, [dashboard?.compliance.practice_days_by_member, sortField, sortDirection]);

  // Filter visible tickets that can be verified (RESOLVED status)
  const verifiableTickets = useMemo(() => {
    if (!dashboard?.drilldown.tickets_visible) return [];
    return dashboard.drilldown.tickets_visible.filter(
      (t) => t.status === "RESOLVED"
    );
  }, [dashboard?.drilldown.tickets_visible]);

  const handleSort = (field: typeof sortField) => {
    if (sortField === field) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const SortIcon = ({ field }: { field: typeof sortField }) => {
    if (sortField !== field) return null;
    return sortDirection === "asc" ? (
      <ChevronUp className="h-4 w-4 inline ml-1" />
    ) : (
      <ChevronDown className="h-4 w-4 inline ml-1" />
    );
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-6 pb-6 space-y-6 max-w-7xl">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
        <TableSkeleton rows={5} />
      </div>
    );
  }

  // Error state
  if (error || !dashboard) {
    return (
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-6 pb-6 space-y-6 max-w-7xl">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Leader Dashboard</h1>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load dashboard data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const { compliance, risk_summary, private_ticket_aggregates, drilldown, cycle } = dashboard;

  return (
    <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-6 pb-6 space-y-6 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Leader Dashboard</h1>
        <p className="text-muted-foreground">
          Team health and compliance overview
          {primaryTeam?.role === "SECTION_LEADER" && primaryTeam.section
            ? ` for ${primaryTeam.section} section`
            : ""}
          {cycle && (
            <span className="ml-2 text-sm">
              · Cycle: {new Date(cycle.date).toLocaleDateString()}
            </span>
          )}
        </p>
      </div>

      {/* Risk Summary Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <RiskCard
          title="Blocking Issues"
          count={risk_summary.blocking_due_count}
          icon={AlertTriangle}
          variant={risk_summary.blocking_due_count >= 3 ? "danger" : risk_summary.blocking_due_count > 0 ? "warning" : "default"}
        />
        <RiskCard
          title="Blocked Members"
          count={risk_summary.blocked_count}
          icon={Clock}
          variant={risk_summary.blocked_count > 0 ? "warning" : "default"}
        />
        <RiskCard
          title="Pending Verification"
          count={risk_summary.resolved_not_verified_count}
          icon={CheckCircle}
          variant={risk_summary.resolved_not_verified_count >= 5 ? "warning" : "default"}
        />
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1 px-4 pt-4">
            <CardTitle className="text-sm font-medium">Total Practice</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="px-4 pb-4 pt-1">
            <div className="text-2xl font-bold">
              {Math.round(compliance.total_practice_minutes_7d / 60)}h {compliance.total_practice_minutes_7d % 60}m
            </div>
            <p className="text-xs text-muted-foreground mt-1">Last 7 days team total</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="compliance" className="space-y-4">
        <TabsList>
          <TabsTrigger value="compliance" className="gap-2">
            <Users className="h-4 w-4" />
            Compliance
          </TabsTrigger>
          <TabsTrigger value="tickets" className="gap-2">
            <CheckCircle className="h-4 w-4" />
            Tickets
          </TabsTrigger>
          <TabsTrigger value="aggregates" className="gap-2">
            <Lock className="h-4 w-4" />
            Private Aggregates
          </TabsTrigger>
        </TabsList>

        {/* Compliance Tab */}
        <TabsContent value="compliance" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-3">
            {/* Radial Progress Card */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Practice Compliance</CardTitle>
                <CardDescription>Members who logged in last 7 days</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center pt-4">
                <ComplianceRadial percentage={compliance.logged_last_7_days_pct} />
                <p className="mt-4 text-sm text-muted-foreground text-center">
                  {sortedMembers.filter((m) => m.days_logged_7d > 0).length} of{" "}
                  {sortedMembers.length} members active
                </p>
              </CardContent>
            </Card>

            {/* Members Table */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Practice Days by Member</CardTitle>
                <CardDescription>Click column headers to sort</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead
                        className="cursor-pointer hover:text-foreground"
                        onClick={() => handleSort("name")}
                      >
                        Name <SortIcon field="name" />
                      </TableHead>
                      <TableHead
                        className="cursor-pointer hover:text-foreground"
                        onClick={() => handleSort("section")}
                      >
                        Section <SortIcon field="section" />
                      </TableHead>
                      <TableHead
                        className="cursor-pointer hover:text-foreground text-right"
                        onClick={() => handleSort("days_logged_7d")}
                      >
                        Days Logged (7d) <SortIcon field="days_logged_7d" />
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sortedMembers.map((member) => (
                      <TableRow key={member.member_id}>
                        <TableCell className="font-medium">{member.name}</TableCell>
                        <TableCell>{member.section || "—"}</TableCell>
                        <TableCell className="text-right">
                          <Badge
                            variant={member.days_logged_7d === 0 ? "destructive" : member.days_logged_7d >= 5 ? "default" : "secondary"}
                          >
                            {member.days_logged_7d}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                    {sortedMembers.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-muted-foreground">
                          No members found
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          <ComplianceInsightsCard
            insights={complianceInsights}
            isLoading={isInsightsLoading}
            error={insightsError instanceof Error ? insightsError : null}
          />

          {/* Risk Breakdown */}
          {(risk_summary.by_section.length > 0 || risk_summary.by_song.length > 0) && (
            <div className="grid gap-4 md:grid-cols-2">
              {risk_summary.by_section.length > 0 && (
                <Card>
                  <CardHeader className="py-4">
                    <CardTitle className="text-base">Risk by Section</CardTitle>
                  </CardHeader>
                  <CardContent className="pb-4">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Section</TableHead>
                          <TableHead className="text-center w-24">Blocking</TableHead>
                          <TableHead className="text-center w-24">Blocked</TableHead>
                          <TableHead className="text-center w-24">Unverified</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {risk_summary.by_section.map((s) => (
                          <TableRow key={s.section}>
                            <TableCell className="font-medium">{s.section}</TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                {s.blocking_due > 0 ? (
                                  <Badge variant="destructive">{s.blocking_due}</Badge>
                                ) : (
                                  <span>0</span>
                                )}
                              </div>
                            </TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                <span>{s.blocked}</span>
                              </div>
                            </TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                <span>{s.resolved_not_verified}</span>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              )}

              {risk_summary.by_song.length > 0 && (
                <Card>
                  <CardHeader className="py-4">
                    <CardTitle className="text-base">Risk by Song</CardTitle>
                  </CardHeader>
                  <CardContent className="pb-4">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Song</TableHead>
                          <TableHead className="text-center w-24">Blocking</TableHead>
                          <TableHead className="text-center w-24">Blocked</TableHead>
                          <TableHead className="text-center w-24">Unverified</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {risk_summary.by_song.map((s) => (
                          <TableRow key={s.song_ref}>
                            <TableCell className="font-medium">{s.song_ref}</TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                {s.blocking_due > 0 ? (
                                  <Badge variant="destructive">{s.blocking_due}</Badge>
                                ) : (
                                  <span>0</span>
                                )}
                              </div>
                            </TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                <span>{s.blocked}</span>
                              </div>
                            </TableCell>
                            <TableCell className="w-24">
                              <div className="flex justify-center items-center">
                                <span>{s.resolved_not_verified}</span>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        {/* Tickets Tab */}
        <TabsContent value="tickets" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-3">
            {/* Members at Risk */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Members at Risk</CardTitle>
                <CardDescription>Open or blocked tickets</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {drilldown.members.filter((m) => m.open_ticket_count > 0 || m.blocked_count > 0).length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No members at risk
                    </p>
                  ) : (
                    drilldown.members
                      .filter((m) => m.open_ticket_count > 0 || m.blocked_count > 0)
                      .sort((a, b) => b.blocked_count - a.blocked_count)
                      .map((member) => (
                        <div
                          key={member.member_id}
                          className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                        >
                          <div>
                            <p className="font-medium text-sm">{member.name}</p>
                            <p className="text-xs text-muted-foreground">{member.section || "No section"}</p>
                          </div>
                          <div className="flex gap-2">
                            {member.blocked_count > 0 && (
                              <Badge variant="destructive">{member.blocked_count} blocked</Badge>
                            )}
                            {member.open_ticket_count > 0 && (
                              <Badge variant="secondary">{member.open_ticket_count} open</Badge>
                            )}
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Visible Tickets */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Tickets Awaiting Verification</CardTitle>
                <CardDescription>
                  {verifiableTickets.length} resolved tickets ready for verification
                </CardDescription>
              </CardHeader>
              <CardContent>
                {verifiableTickets.length === 0 ? (
                  <NoTicketsToVerifyEmptyState />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Title</TableHead>
                        <TableHead>Priority</TableHead>
                        <TableHead>Section</TableHead>
                        <TableHead className="text-right">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {verifiableTickets.map((ticket) => (
                        <TableRow key={ticket.id}>
                          <TableCell className="font-medium max-w-[200px] truncate">
                            {ticket.title}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={
                                ticket.priority === "BLOCKING"
                                  ? "destructive"
                                  : ticket.priority === "MEDIUM"
                                    ? "default"
                                    : "secondary"
                              }
                            >
                              {ticket.priority}
                            </Badge>
                          </TableCell>
                          <TableCell>{ticket.section || "—"}</TableCell>
                          <TableCell className="text-right">
                            <Button
                              size="sm"
                              onClick={() => setSelectedTicket(ticket)}
                            >
                              <CheckCircle className="h-4 w-4 mr-1" />
                              Verify
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Private Aggregates Tab */}
        <TabsContent value="aggregates" className="space-y-4">
          <Alert>
            <Shield className="h-4 w-4" />
            <AlertDescription>
              <strong>Privacy Notice:</strong> This data shows aggregate counts only.
              No ticket IDs, owner names, or other identifying information is displayed
              to protect member privacy.
            </AlertDescription>
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lock className="h-5 w-5" />
                Anonymous Ticket Aggregates
              </CardTitle>
              <CardDescription>
                Private ticket data aggregated by category, status, and priority
              </CardDescription>
            </CardHeader>
            <CardContent>
              {private_ticket_aggregates.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No private ticket data available
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Section</TableHead>
                      <TableHead>Category</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Priority</TableHead>
                      <TableHead>Due</TableHead>
                      <TableHead className="text-right">Count</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {private_ticket_aggregates.map((agg, i) => (
                      <TableRow key={i}>
                        <TableCell>{agg.section || "—"}</TableCell>
                        <TableCell>{agg.category}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{agg.status}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              agg.priority === "BLOCKING"
                                ? "destructive"
                                : agg.priority === "MEDIUM"
                                  ? "default"
                                  : "secondary"
                            }
                          >
                            {agg.priority}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs">
                          {agg.due_bucket.replace(/_/g, " ")}
                        </TableCell>
                        <TableCell className="text-right font-bold">{agg.count}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Verify Ticket Dialog */}
      <Dialog open={!!selectedTicket} onOpenChange={(open) => !open && setSelectedTicket(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Verify Ticket</DialogTitle>
            <DialogDescription>
              Confirm that this issue has been resolved satisfactorily.
            </DialogDescription>
          </DialogHeader>

          {selectedTicket && (
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-muted">
                <p className="font-medium">{selectedTicket.title}</p>
                <div className="flex gap-2 mt-2">
                  <Badge variant={selectedTicket.priority === "BLOCKING" ? "destructive" : "secondary"}>
                    {selectedTicket.priority}
                  </Badge>
                  <Badge variant="outline">{selectedTicket.section || "No section"}</Badge>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="verify-note">Verification Note (optional)</Label>
                <Textarea
                  id="verify-note"
                  placeholder="Add any feedback or notes about the resolution..."
                  value={verifyNote}
                  onChange={(e) => setVerifyNote(e.target.value)}
                  rows={3}
                />
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedTicket(null)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (selectedTicket) {
                  verifyMutation.mutate({
                    ticketId: selectedTicket.id,
                    content: verifyNote || undefined,
                  });
                }
              }}
              disabled={verifyMutation.isPending}
            >
              {verifyMutation.isPending ? (
                <>
                  <ButtonSpinner className="mr-2" />
                  Verifying...
                </>
              ) : (
                "Verify Ticket"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
