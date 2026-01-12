/**
 * Ticket Detail Page
 *
 * Shows ticket details, metadata, activity timeline, and transition actions.
 * Data source: Ticket data passed via navigation state (no GET /tickets/{id} endpoint).
 * Activity is fetched separately.
 */

import { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import {
  getTicketActivity,
  transitionTicket,
  verifyTicket,
} from "@/lib/api/client";
import type {
  TicketResponse,
  TicketActivityResponse,
  TicketStatus,
} from "@/lib/api/types";
import { useAuth } from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Avatar } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Globe,
  Lock,
  Users,
} from "lucide-react";
import { format, parseISO } from "date-fns";

// Reuse configurations
const priorityConfig = {
  BLOCKING: {
    label: "Blocking",
    className: "bg-red-500/15 text-red-600 border-red-500/30",
  },
  MEDIUM: {
    label: "Medium",
    className: "bg-amber-500/15 text-amber-600 border-amber-500/30",
  },
  LOW: {
    label: "Low",
    className: "bg-slate-500/15 text-slate-600 border-slate-500/30",
  },
} as const;

const statusConfig = {
  OPEN: {
    label: "Open",
    icon: <AlertCircle className="h-3 w-3" />,
    className: "bg-blue-500/15 text-blue-600",
  },
  IN_PROGRESS: {
    label: "In Progress",
    icon: <Clock className="h-3 w-3" />,
    className: "bg-violet-500/15 text-violet-600",
  },
  BLOCKED: {
    label: "Blocked",
    icon: <AlertCircle className="h-3 w-3" />,
    className: "bg-red-500/15 text-red-600",
  },
  RESOLVED: {
    label: "Resolved",
    icon: <CheckCircle2 className="h-3 w-3" />,
    className: "bg-emerald-500/15 text-emerald-600",
  },
  VERIFIED: {
    label: "Verified",
    icon: <CheckCircle2 className="h-3 w-3" />,
    className: "bg-green-500/15 text-green-600",
  },
} as const;

const visibilityIcons = {
  PRIVATE: <Lock className="h-3 w-3" />,
  SECTION: <Users className="h-3 w-3" />,
  TEAM: <Globe className="h-3 w-3" />,
} as const;

const categoryLabels = {
  PITCH: "Pitch",
  RHYTHM: "Rhythm",
  MEMORY: "Memory",
  BLEND: "Blend",
  TECHNIQUE: "Technique",
  OTHER: "Other",
} as const;

export function TicketDetailPage() {
  const { ticketId } = useParams<{ ticketId: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const { user, primaryTeam } = useAuth();

  // Ticket data from navigation state
  const [ticket, setTicket] = useState<TicketResponse | null>(
    location.state?.ticket || null
  );
  const [activities, setActivities] = useState<TicketActivityResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Transition modal state
  const [showResolveModal, setShowResolveModal] = useState(false);
  const [showVerifyModal, setShowVerifyModal] = useState(false);
  const [modalNote, setModalNote] = useState("");
  const [transitioning, setTransitioning] = useState(false);

  // Fetch activity timeline
  useEffect(() => {
    if (!ticketId) return;

    const fetchActivity = async () => {
      try {
        const response = await getTicketActivity(ticketId);
        setActivities(response.items);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load activity");
      } finally {
        setLoading(false);
      }
    };

    fetchActivity();
  }, [ticketId]);

  const handleTransition = async (toStatus: TicketStatus, content?: string) => {
    if (!ticketId) return;

    setTransitioning(true);
    try {
      const response = await transitionTicket(ticketId, {
        to_status: toStatus,
        content,
      });
      setTicket(response.ticket);
      setShowResolveModal(false);
      setModalNote("");
      // Refresh activity
      const activityResponse = await getTicketActivity(ticketId);
      setActivities(activityResponse.items);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Transition failed");
    } finally {
      setTransitioning(false);
    }
  };

  const handleVerify = async () => {
    if (!ticketId) return;

    setTransitioning(true);
    try {
      const response = await verifyTicket(ticketId, {
        content: modalNote || undefined,
      });
      setTicket(response.ticket);
      setShowVerifyModal(false);
      setModalNote("");
      // Refresh activity
      const activityResponse = await getTicketActivity(ticketId);
      setActivities(activityResponse.items);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setTransitioning(false);
    }
  };

  const canVerify = () => {
    if (!primaryTeam || !ticket) return false;
    const role = primaryTeam.role;
    const userSection = primaryTeam.section;

    if (role === "ADMIN") return true;
    if (role === "SECTION_LEADER") {
      if (ticket.visibility === "TEAM") return true;
      if (ticket.visibility === "SECTION" && ticket.section === userSection)
        return true;
      if (ticket.visibility === "PRIVATE" && ticket.section === userSection)
        return true;
    }
    return false;
  };

  const isOwner = ticket && user && ticket.owner_id === user.id;

  if (!ticket) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" onClick={() => navigate("/tickets")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Tickets
        </Button>
        <Card>
          <CardContent className="p-6">
            <p className="text-center text-muted-foreground">
              Ticket not found. Please navigate from the tickets list.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const status = statusConfig[ticket.status];
  const priority = priorityConfig[ticket.priority];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Button variant="ghost" onClick={() => navigate("/tickets")} className="mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Tickets
        </Button>

        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-muted-foreground">
                {visibilityIcons[ticket.visibility]}
              </span>
              <h1 className="text-3xl font-bold tracking-tight">{ticket.title}</h1>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="outline"
                className={cn("flex items-center gap-1", status.className)}
              >
                {status.icon}
                {status.label}
              </Badge>
              <Badge variant="outline" className={priority.className}>
                {priority.label}
              </Badge>
            </div>
          </div>

          {ticket.due_at && (
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Due date</p>
              <p className="text-lg font-semibold">
                {format(parseISO(ticket.due_at), "MMM d, yyyy")}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Action Bar */}
      {isOwner && (
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-2">
              {ticket.status === "OPEN" && (
                <Button onClick={() => handleTransition("IN_PROGRESS")}>
                  Start Working
                </Button>
              )}

              {ticket.status === "IN_PROGRESS" && (
                <>
                  <Button
                    variant="outline"
                    onClick={() => handleTransition("BLOCKED")}
                  >
                    Mark Blocked
                  </Button>
                  <Button onClick={() => setShowResolveModal(true)}>
                    Resolve
                  </Button>
                </>
              )}

              {ticket.status === "BLOCKED" && (
                <>
                  <Button onClick={() => handleTransition("IN_PROGRESS")}>
                    Resume
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowResolveModal(true)}
                  >
                    Resolve
                  </Button>
                </>
              )}

              {ticket.status === "RESOLVED" && canVerify() && (
                <Button onClick={() => setShowVerifyModal(true)}>
                  Verify
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-3">
        {/* Main content - 2 columns */}
        <div className="md:col-span-2 space-y-6">
          {/* Description */}
          <Card>
            <CardHeader>
              <CardTitle>Description</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground whitespace-pre-wrap">
                {ticket.description || "No description provided."}
              </p>
            </CardContent>
          </Card>

          {/* Activity Timeline */}
          <Card>
            <CardHeader>
              <CardTitle>Activity Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <p className="text-center text-muted-foreground">Loading activity...</p>
              ) : error ? (
                <p className="text-center text-red-600">{error}</p>
              ) : activities.length === 0 ? (
                <p className="text-center text-muted-foreground">No activity yet.</p>
              ) : (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-4">
                    {activities.map((activity, index) => (
                      <div key={activity.id}>
                        <div className="flex gap-3">
                          <Avatar className="h-8 w-8 bg-muted">
                            <span className="text-xs">
                              {activity.user_id.slice(0, 2).toUpperCase()}
                            </span>
                          </Avatar>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 text-sm">
                              <span className="font-medium">User</span>
                              <span className="text-muted-foreground">
                                {format(
                                  parseISO(activity.created_at),
                                  "MMM d, h:mm a"
                                )}
                              </span>
                            </div>

                            {activity.type === "STATUS_CHANGE" &&
                              activity.old_status &&
                              activity.new_status && (
                                <p className="text-sm text-muted-foreground mt-1">
                                  Status: {activity.old_status} → {activity.new_status}
                                </p>
                              )}

                            {activity.type === "VERIFIED" && (
                              <p className="text-sm text-emerald-600 mt-1 font-medium">
                                Verified ticket
                              </p>
                            )}

                            {activity.type === "CLAIMED" && (
                              <p className="text-sm text-blue-600 mt-1 font-medium">
                                Claimed ticket
                              </p>
                            )}

                            {activity.type === "CREATED" && (
                              <p className="text-sm text-muted-foreground mt-1">
                                Created ticket
                              </p>
                            )}

                            {activity.content && (
                              <p className="text-sm mt-2 p-3 bg-muted rounded-md">
                                {activity.content}
                              </p>
                            )}
                          </div>
                        </div>
                        {index < activities.length - 1 && <Separator className="mt-4" />}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Metadata Sidebar - 1 column */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Metadata</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div>
                <p className="text-muted-foreground">Category</p>
                <p className="font-medium">{categoryLabels[ticket.category]}</p>
              </div>

              {ticket.song_ref && (
                <div>
                  <p className="text-muted-foreground">Song</p>
                  <p className="font-medium">{ticket.song_ref}</p>
                </div>
              )}

              {ticket.section && (
                <div>
                  <p className="text-muted-foreground">Section</p>
                  <p className="font-medium">{ticket.section}</p>
                </div>
              )}

              <Separator />

              <div>
                <p className="text-muted-foreground">Created by</p>
                <p className="font-medium">{ticket.created_by}</p>
              </div>

              {ticket.owner_id && (
                <div>
                  <p className="text-muted-foreground">Owner</p>
                  <p className="font-medium">{ticket.owner_id}</p>
                </div>
              )}

              {ticket.claimed_by && (
                <div>
                  <p className="text-muted-foreground">Claimed by</p>
                  <p className="font-medium">{ticket.claimed_by}</p>
                </div>
              )}

              {ticket.resolved_at && (
                <div>
                  <p className="text-muted-foreground">Resolved at</p>
                  <p className="font-medium">
                    {format(parseISO(ticket.resolved_at), "MMM d, yyyy")}
                  </p>
                </div>
              )}

              {ticket.verified_by && (
                <div>
                  <p className="text-muted-foreground">Verified by</p>
                  <p className="font-medium">{ticket.verified_by}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Resolve Modal */}
      <Dialog open={showResolveModal} onOpenChange={setShowResolveModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Resolve Ticket</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Please provide a note describing how this issue was resolved.
            </p>
            <textarea
              className="w-full min-h-[100px] p-3 rounded-md border bg-background resize-none"
              placeholder="Describe what you did to resolve this issue..."
              value={modalNote}
              onChange={(e) => setModalNote(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowResolveModal(false);
                setModalNote("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={() => handleTransition("RESOLVED", modalNote)}
              disabled={!modalNote.trim() || transitioning}
            >
              {transitioning ? "Resolving..." : "Mark as Resolved"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Verify Modal */}
      <Dialog open={showVerifyModal} onOpenChange={setShowVerifyModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Verify Ticket</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Add an optional note with your verification.
            </p>
            <textarea
              className="w-full min-h-[100px] p-3 rounded-md border bg-background resize-none"
              placeholder="Add verification notes (optional)..."
              value={modalNote}
              onChange={(e) => setModalNote(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowVerifyModal(false);
                setModalNote("");
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleVerify} disabled={transitioning}>
              {transitioning ? "Verifying..." : "Verify"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
