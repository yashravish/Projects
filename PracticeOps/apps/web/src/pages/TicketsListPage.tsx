/**
 * Tickets List Page
 *
 * Clean list view of tickets with minimal filtering.
 */

import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  listTickets,
  claimTicket,
  getActiveCycle,
  type ListTicketsFilters,
} from "@/lib/api/client";
import type { TicketResponse, TicketStatus } from "@/lib/api/types";
import { useAuth } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { ListSkeleton } from "@/components/ui/loading";
import { NoCycleEmptyState } from "@/components/ui/empty-state";
import { toastSuccess, toastError } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { format, parseISO } from "date-fns";

export function TicketsListPage() {
  const { primaryTeam } = useAuth();

  const [tickets, setTickets] = useState<TicketResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [cycleId, setCycleId] = useState<string | null>(null);
  const [noCycle, setNoCycle] = useState(false);
  const [filter, setFilter] = useState<"all" | "open" | "mine">("all");
  const [claiming, setClaiming] = useState<string | null>(null);

  // Load active cycle
  useEffect(() => {
    if (!primaryTeam?.team_id) return;

    getActiveCycle(primaryTeam.team_id)
      .then((res) => {
        if (res.cycle) {
          setCycleId(res.cycle.id);
        } else {
          setNoCycle(true);
          setLoading(false);
        }
      })
      .catch(() => {
        setNoCycle(true);
        setLoading(false);
      });
  }, [primaryTeam?.team_id]);

  // Load tickets
  useEffect(() => {
    if (!cycleId) return;

    const filters: ListTicketsFilters = {};
    if (filter === "open") {
      filters.status = "OPEN";
    }
    // Note: "mine" filter is not supported by the API yet

    setLoading(true);
    listTickets(cycleId, filters)
      .then((res) => {
        setTickets(res.items);
      })
      .catch(() => {
        toastError("Failed to load tickets");
      })
      .finally(() => setLoading(false));
  }, [cycleId, filter]);

  const handleClaim = async (ticketId: string) => {
    setClaiming(ticketId);
    try {
      await claimTicket(ticketId);
      toastSuccess("Ticket claimed");
      // Refresh
      if (cycleId) {
        const res = await listTickets(cycleId);
        setTickets(res.items);
      }
    } catch (err) {
      toastError("Failed to claim ticket");
    } finally {
      setClaiming(null);
    }
  };

  if (noCycle) {
    return (
      <div className="p-6">
        <NoCycleEmptyState />
      </div>
    );
  }

  if (loading && tickets.length === 0) {
    return (
      <div className="p-6">
        <ListSkeleton />
      </div>
    );
  }

  const statusOrder: TicketStatus[] = ["BLOCKED", "OPEN", "IN_PROGRESS", "RESOLVED", "VERIFIED"];
  const sortedTickets = [...tickets].sort((a, b) => {
    // Blocking first
    if (a.priority === "BLOCKING" && b.priority !== "BLOCKING") return -1;
    if (b.priority === "BLOCKING" && a.priority !== "BLOCKING") return 1;
    // Then by status
    return statusOrder.indexOf(a.status) - statusOrder.indexOf(b.status);
  });

  return (
    <div className="min-h-full bg-neutral-50">
      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-xl font-semibold text-neutral-900">Tickets</h1>
          <div className="flex items-center gap-1 text-sm">
            {(["all", "open", "mine"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "px-3 py-1.5 rounded transition-colors",
                  filter === f
                    ? "bg-white text-neutral-900 shadow-sm"
                    : "text-neutral-500 hover:text-neutral-700"
                )}
              >
                {f === "all" ? "All" : f === "open" ? "Open" : "Mine"}
              </button>
            ))}
          </div>
        </div>

        {/* Tickets list */}
        {sortedTickets.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-neutral-400 mb-2">No tickets</div>
            <p className="text-sm text-neutral-500">
              {filter !== "all" ? "Try changing the filter" : "Nothing to resolve"}
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {sortedTickets.map((ticket) => (
              <div
                key={ticket.id}
                className={cn(
                  "flex items-start justify-between py-4 px-4 -mx-4 rounded-lg transition-colors",
                  ticket.priority === "BLOCKING" && ticket.status !== "VERIFIED"
                    ? "bg-red-50"
                    : "hover:bg-white"
                )}
              >
                <Link
                  to={`/tickets/${ticket.id}`}
                  className="flex-1 min-w-0"
                >
                  <div className="flex items-start gap-3">
                    <span
                      className={cn(
                        "w-2 h-2 rounded-full mt-2 flex-shrink-0",
                        ticket.status === "VERIFIED" ? "bg-green-500" :
                        ticket.status === "RESOLVED" ? "bg-emerald-500" :
                        ticket.status === "BLOCKED" ? "bg-red-500" :
                        ticket.status === "IN_PROGRESS" ? "bg-blue-500" :
                        ticket.priority === "BLOCKING" ? "bg-red-500" :
                        "bg-neutral-300"
                      )}
                    />
                    <div className="min-w-0">
                      <div className="text-sm text-neutral-900 truncate">
                        {ticket.title}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        {ticket.song_ref && (
                          <span className="text-xs text-neutral-500">
                            {ticket.song_ref}
                          </span>
                        )}
                        {ticket.section && (
                          <span className="text-xs text-neutral-400">
                            {ticket.section}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </Link>

                <div className="flex items-center gap-4 ml-4">
                  <span className="text-xs text-neutral-400 whitespace-nowrap">
                    {ticket.status === "VERIFIED" ? "Verified" :
                     ticket.status === "RESOLVED" ? "Resolved" :
                     ticket.status === "BLOCKED" ? "Blocked" :
                     ticket.status === "IN_PROGRESS" ? "In progress" :
                     "Open"}
                  </span>
                  
                  {ticket.due_at && (
                    <span className="text-xs text-neutral-400 tabular-nums whitespace-nowrap">
                      {format(parseISO(ticket.due_at), "MMM d")}
                    </span>
                  )}

                  {ticket.status === "OPEN" && ticket.claimable && !ticket.owner_id && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.preventDefault();
                        handleClaim(ticket.id);
                      }}
                      disabled={claiming === ticket.id}
                      className="h-7 text-xs"
                    >
                      {claiming === ticket.id ? "..." : "Claim"}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
