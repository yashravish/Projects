/**
 * Tickets Due Soon List - Shows tickets that need attention
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Ticket, CheckCircle2, AlertCircle, Clock, Lock, Users, Globe } from "lucide-react";
import { format, parseISO, isPast, isToday } from "date-fns";
import type { TicketDueSoon, Priority, TicketStatus, TicketVisibility } from "@/lib/api/types";

interface TicketsListProps {
  tickets: TicketDueSoon[];
  maxHeight?: string;
}

const priorityConfig: Record<Priority, { label: string; className: string }> = {
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
};

const statusConfig: Record<TicketStatus, { label: string; icon: React.ReactNode; className: string }> = {
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
};

const visibilityIcons: Record<TicketVisibility, React.ReactNode> = {
  PRIVATE: <Lock className="h-3 w-3" />,
  SECTION: <Users className="h-3 w-3" />,
  TEAM: <Globe className="h-3 w-3" />,
};

function formatDueDate(dateStr: string | null): { text: string; isOverdue: boolean; isToday: boolean } {
  if (!dateStr) return { text: "No due date", isOverdue: false, isToday: false };
  
  const date = parseISO(dateStr);
  const overdue = isPast(date) && !isToday(date);
  const today = isToday(date);
  
  return {
    text: today ? "Due today" : overdue ? `Overdue: ${format(date, "MMM d")}` : format(date, "MMM d"),
    isOverdue: overdue,
    isToday: today,
  };
}

export function TicketsList({ tickets, maxHeight = "280px" }: TicketsListProps) {
  if (tickets.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Ticket className="h-4 w-4 text-muted-foreground" />
            Tickets Due Soon
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/10 mb-3">
              <CheckCircle2 className="h-6 w-6 text-emerald-500" />
            </div>
            <p className="text-sm font-medium text-emerald-600">All caught up!</p>
            <p className="text-xs text-muted-foreground mt-1">
              No tickets due soon
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Ticket className="h-4 w-4 text-muted-foreground" />
            Tickets Due Soon
            <Badge variant="secondary" className="ml-1 text-xs">
              {tickets.length}
            </Badge>
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea style={{ maxHeight }}>
          <div className="divide-y">
            {tickets.map((ticket) => {
              const dueInfo = formatDueDate(ticket.due_at);
              const status = statusConfig[ticket.status];
              
              return (
                <div
                  key={ticket.id}
                  className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-muted/50"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">
                        {visibilityIcons[ticket.visibility]}
                      </span>
                      <p className="text-sm font-medium truncate">
                        {ticket.title}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge 
                        variant="outline" 
                        className={cn("text-xs py-0 h-5 flex items-center gap-1", status.className)}
                      >
                        {status.icon}
                        {status.label}
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={cn("text-xs py-0 h-5", priorityConfig[ticket.priority].className)}
                      >
                        {priorityConfig[ticket.priority].label}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <p className={cn(
                      "text-xs",
                      dueInfo.isOverdue && "text-red-600 font-medium",
                      dueInfo.isToday && "text-amber-600 font-medium",
                      !dueInfo.isOverdue && !dueInfo.isToday && "text-muted-foreground"
                    )}>
                      {dueInfo.text}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

