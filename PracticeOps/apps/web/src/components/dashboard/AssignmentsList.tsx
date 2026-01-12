/**
 * Assignments List - Shows current cycle assignments sorted by priority
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Music2, BookOpen, Headphones, Brain, ClipboardList } from "lucide-react";
import { format, parseISO, isPast, isToday } from "date-fns";
import type { AssignmentSummary, Priority, AssignmentType } from "@/lib/api/types";

interface AssignmentsListProps {
  assignments: AssignmentSummary[];
  maxHeight?: string;
}

const priorityConfig: Record<Priority, { label: string; className: string }> = {
  BLOCKING: {
    label: "Blocking",
    className: "bg-red-500/15 text-red-600 border-red-500/30 hover:bg-red-500/20",
  },
  MEDIUM: {
    label: "Medium",
    className: "bg-amber-500/15 text-amber-600 border-amber-500/30 hover:bg-amber-500/20",
  },
  LOW: {
    label: "Low",
    className: "bg-slate-500/15 text-slate-600 border-slate-500/30 hover:bg-slate-500/20",
  },
};

const typeIcons: Record<AssignmentType, React.ReactNode> = {
  SONG_WORK: <Music2 className="h-4 w-4" />,
  TECHNIQUE: <BookOpen className="h-4 w-4" />,
  LISTENING: <Headphones className="h-4 w-4" />,
  MEMORIZATION: <Brain className="h-4 w-4" />,
};

function formatDueDate(dateStr: string | null): { text: string; isOverdue: boolean; isToday: boolean } {
  if (!dateStr) return { text: "No due date", isOverdue: false, isToday: false };
  
  const date = parseISO(dateStr);
  const overdue = isPast(date) && !isToday(date);
  const today = isToday(date);
  
  return {
    text: today ? "Due today" : format(date, "MMM d"),
    isOverdue: overdue,
    isToday: today,
  };
}

export function AssignmentsList({ assignments, maxHeight = "320px" }: AssignmentsListProps) {
  // Sort by priority (BLOCKING first) then by due_at
  const sortedAssignments = [...assignments].sort((a, b) => {
    const priorityOrder: Record<Priority, number> = { BLOCKING: 0, MEDIUM: 1, LOW: 2 };
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (priorityDiff !== 0) return priorityDiff;
    
    // Sort by due date (nulls last)
    if (!a.due_at && !b.due_at) return 0;
    if (!a.due_at) return 1;
    if (!b.due_at) return -1;
    return new Date(a.due_at).getTime() - new Date(b.due_at).getTime();
  });

  if (assignments.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-muted-foreground" />
            Assignments
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted mb-3">
              <ClipboardList className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm font-medium">No assignments</p>
            <p className="text-xs text-muted-foreground mt-1">
              No assignments for this cycle yet
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const blockingCount = assignments.filter((a) => a.priority === "BLOCKING").length;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-muted-foreground" />
            Assignments
            <Badge variant="secondary" className="ml-1 text-xs">
              {assignments.length}
            </Badge>
          </CardTitle>
          {blockingCount > 0 && (
            <Badge className={priorityConfig.BLOCKING.className}>
              {blockingCount} blocking
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea style={{ maxHeight }}>
          <div className="divide-y">
            {sortedAssignments.map((assignment) => {
              const dueInfo = formatDueDate(assignment.due_at);
              const isBlocking = assignment.priority === "BLOCKING";
              
              return (
                <div
                  key={assignment.id}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 transition-colors hover:bg-muted/50",
                    isBlocking && "bg-red-500/5"
                  )}
                >
                  <div className={cn(
                    "flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
                    isBlocking ? "bg-red-500/15 text-red-600" : "bg-muted text-muted-foreground"
                  )}>
                    {typeIcons[assignment.type]}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className={cn(
                      "text-sm font-medium truncate",
                      isBlocking && "text-red-600"
                    )}>
                      {assignment.title}
                    </p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <Badge 
                        variant="outline" 
                        className={cn("text-xs py-0 h-5", priorityConfig[assignment.priority].className)}
                      >
                        {priorityConfig[assignment.priority].label}
                      </Badge>
                      {assignment.section && (
                        <span className="text-xs text-muted-foreground">
                          {assignment.section}
                        </span>
                      )}
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

