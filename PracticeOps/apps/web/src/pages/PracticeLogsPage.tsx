/**
 * Practice Logs Page
 *
 * Displays user's practice logs with ability to log new sessions.
 * Uses CONTRACT LOCK protocol - client.ts wrappers only.
 */

import { useState, useEffect } from "react";
import { format, parseISO, formatDistanceToNow } from "date-fns";
import {
  listPracticeLogs,
  getActiveCycle,
  createPracticeLog,
  type ListPracticeLogsFilters,
} from "@/lib/api/client";
import type { PracticeLogResponse, CreatePracticeLogRequest } from "@/lib/api/types";
import { useAuth } from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { ListSkeleton } from "@/components/ui/loading";
import {
  NoPracticeLogsEmptyState,
  NoCycleEmptyState,
} from "@/components/ui/empty-state";
import { toastSuccess, toastError } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Plus,
  Clock,
  Calendar,
  Star,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";

export function PracticeLogsPage() {
  const { primaryTeam } = useAuth();
  const [logs, setLogs] = useState<PracticeLogResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [cycleId, setCycleId] = useState<string | null>(null);

  // New log dialog state
  const [dialogOpen, setDialogOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [newLog, setNewLog] = useState<CreatePracticeLogRequest>({
    duration_min: 30,
    notes: "",
    rating_1_5: 3,
    blocked_flag: false,
    assignment_ids: [],
  });

  // Load active cycle
  useEffect(() => {
    const fetchActiveCycle = async () => {
      const teamId = primaryTeam?.team_id;
      if (!teamId) return;

      try {
        const response = await getActiveCycle(teamId);
        if (response.cycle) {
          setCycleId(response.cycle.id);
        }
      } catch (err) {
        console.error("Failed to fetch active cycle:", err);
      }
    };

    fetchActiveCycle();
  }, [primaryTeam]);

  // Fetch practice logs
  useEffect(() => {
    if (!cycleId) {
      setLoading(false);
      return;
    }

    const fetchLogs = async () => {
      setLoading(true);
      setError(null);
      try {
        const filters: ListPracticeLogsFilters = { me: true };
        const response = await listPracticeLogs(cycleId, filters);
        setLogs(response.items);
        setNextCursor(response.next_cursor);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load practice logs");
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [cycleId]);

  const handleLoadMore = async () => {
    if (!cycleId || !nextCursor) return;

    try {
      const filters: ListPracticeLogsFilters = { me: true };
      const response = await listPracticeLogs(cycleId, filters, nextCursor);
      setLogs((prev) => [...prev, ...response.items]);
      setNextCursor(response.next_cursor);
    } catch (err) {
      toastError(err instanceof Error ? err.message : "Failed to load more logs");
    }
  };

  const handleCreateLog = async () => {
    if (!cycleId) return;

    setSubmitting(true);
    try {
      const response = await createPracticeLog(cycleId, newLog);
      setLogs((prev) => [response.practice_log, ...prev]);
      setDialogOpen(false);
      setNewLog({
        duration_min: 30,
        notes: "",
        rating_1_5: 3,
        blocked_flag: false,
        assignment_ids: [],
      });
      toastSuccess("Practice logged!");

      // Show suggested ticket prompt if blocked
      if (response.suggested_ticket) {
        toastSuccess(
          "Consider creating a ticket for your blocker",
        );
      }
    } catch (err) {
      toastError(err instanceof Error ? err.message : "Failed to log practice");
    } finally {
      setSubmitting(false);
    }
  };

  const getRatingStars = (rating: number | null) => {
    if (!rating) return null;
    return (
      <div className="flex items-center gap-0.5">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={cn(
              "h-3 w-3",
              star <= rating
                ? "fill-amber-400 text-amber-400"
                : "text-muted-foreground/30"
            )}
          />
        ))}
      </div>
    );
  };

  const LogCard = ({ log }: { log: PracticeLogResponse }) => (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {log.duration_minutes} min
              </Badge>
              {log.rating_1_5 && getRatingStars(log.rating_1_5)}
              {log.blocked_flag && (
                <Badge variant="destructive" className="flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Blocked
                </Badge>
              )}
            </div>

            {log.notes && (
              <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                {log.notes}
              </p>
            )}

            {log.assignments.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {log.assignments.map((a) => (
                  <Badge key={a.id} variant="secondary" className="text-xs">
                    {a.title}
                  </Badge>
                ))}
              </div>
            )}

            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              <span>
                {format(parseISO(log.occurred_at), "MMM d, yyyy 'at' h:mm a")}
              </span>
              <span className="text-muted-foreground/50">•</span>
              <span>{formatDistanceToNow(parseISO(log.occurred_at), { addSuffix: true })}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  // Calculate weekly stats
  const weeklyStats = {
    totalSessions: logs.length,
    totalMinutes: logs.reduce((sum, log) => sum + log.duration_minutes, 0),
    avgRating:
      logs.filter((l) => l.rating_1_5).length > 0
        ? logs
            .filter((l) => l.rating_1_5)
            .reduce((sum, l) => sum + (l.rating_1_5 || 0), 0) /
          logs.filter((l) => l.rating_1_5).length
        : 0,
  };

  if (!cycleId && !loading) {
    return (
      <div className="min-h-full bg-neutral-50">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">Practice Logs</h1>
                <p className="text-muted-foreground">Track your practice sessions</p>
              </div>
            </div>
            <NoCycleEmptyState />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-neutral-50">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Practice Logs</h1>
              <p className="text-muted-foreground">Track your practice sessions</p>
            </div>

        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Log Practice
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Log Practice Session</DialogTitle>
            </DialogHeader>

            <div className="space-y-4 py-4">
              {/* Duration */}
              <div className="space-y-2">
                <Label htmlFor="duration">Duration (minutes)</Label>
                <Input
                  id="duration"
                  type="number"
                  min={1}
                  max={600}
                  value={newLog.duration_min}
                  onChange={(e) =>
                    setNewLog((prev) => ({
                      ...prev,
                      duration_min: parseInt(e.target.value) || 30,
                    }))
                  }
                />
              </div>

              {/* Rating */}
              <div className="space-y-2">
                <Label>How did it go?</Label>
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      className="p-1 hover:scale-110 transition-transform"
                      onClick={() =>
                        setNewLog((prev) => ({ ...prev, rating_1_5: star }))
                      }
                    >
                      <Star
                        className={cn(
                          "h-6 w-6",
                          star <= (newLog.rating_1_5 || 0)
                            ? "fill-amber-400 text-amber-400"
                            : "text-muted-foreground/30 hover:text-amber-400/50"
                        )}
                      />
                    </button>
                  ))}
                </div>
              </div>

              {/* Notes */}
              <div className="space-y-2">
                <Label htmlFor="notes">Notes (optional)</Label>
                <Textarea
                  id="notes"
                  placeholder="What did you work on?"
                  value={newLog.notes || ""}
                  onChange={(e) =>
                    setNewLog((prev) => ({ ...prev, notes: e.target.value }))
                  }
                />
              </div>

              {/* Blocked flag */}
              <div className="flex items-center justify-between rounded-lg border p-3">
                <div className="space-y-0.5">
                  <Label className="text-base">Blocked?</Label>
                  <p className="text-sm text-muted-foreground">
                    Did you encounter any issues?
                  </p>
                </div>
                <Switch
                  checked={newLog.blocked_flag || false}
                  onCheckedChange={(checked) =>
                    setNewLog((prev) => ({ ...prev, blocked_flag: checked }))
                  }
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setDialogOpen(false)}
                disabled={submitting}
              >
                Cancel
              </Button>
              <Button onClick={handleCreateLog} disabled={submitting}>
                {submitting ? "Saving..." : "Log Practice"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats cards */}
      {logs.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Sessions</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{weeklyStats.totalSessions}</div>
              <p className="text-xs text-muted-foreground">this cycle</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Time</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {Math.floor(weeklyStats.totalMinutes / 60)}h {weeklyStats.totalMinutes % 60}m
              </div>
              <p className="text-xs text-muted-foreground">practiced</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Rating</CardTitle>
              <Star className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold flex items-center gap-2">
                {weeklyStats.avgRating.toFixed(1)}
                {getRatingStars(Math.round(weeklyStats.avgRating))}
              </div>
              <p className="text-xs text-muted-foreground">session quality</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <ListSkeleton count={5} variant="card" />
      ) : error ? (
        <Card>
          <CardContent className="p-6">
            <p className="text-center text-red-600">{error}</p>
          </CardContent>
        </Card>
      ) : logs.length === 0 ? (
        <NoPracticeLogsEmptyState onAction={() => setDialogOpen(true)} />
      ) : (
        <>
          {/* Logs list */}
          <div className="space-y-3">
            {logs.map((log) => (
              <LogCard key={log.id} log={log} />
            ))}
          </div>

          {/* Load more */}
          {nextCursor && (
            <div className="flex justify-center">
              <Button onClick={handleLoadMore} variant="outline">
                Load more
              </Button>
            </div>
          )}
        </>
      )}
        </div>
      </div>
    </div>
  );
}
