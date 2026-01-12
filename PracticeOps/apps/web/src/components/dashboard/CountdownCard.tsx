/**
 * Countdown Card - Shows days until next rehearsal
 */

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Calendar, AlertTriangle, CheckCircle2 } from "lucide-react";

interface CountdownCardProps {
  days: number | null;
  cycleLabel: string | null;
  cycleDate?: string | null;  // Optional, for future use
}

export function CountdownCard({ days, cycleLabel, cycleDate: _cycleDate }: CountdownCardProps) {
  // No active cycle
  if (days === null || cycleLabel === null) {
    return (
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
              <Calendar className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Next Rehearsal
              </p>
              <p className="text-lg font-semibold text-muted-foreground">
                Not scheduled
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Determine urgency styling
  const isOverdue = days < 0;
  const isToday = days === 0;
  const isUrgent = days > 0 && days <= 2;

  const getVariant = () => {
    if (isOverdue) return "overdue";
    if (isToday) return "today";
    if (isUrgent) return "urgent";
    return "normal";
  };

  const variant = getVariant();

  const styles = {
    overdue: {
      card: "bg-gradient-to-br from-red-500/10 to-rose-500/10 border-red-500/30",
      icon: "bg-red-500/20 text-red-600",
      text: "text-red-600",
    },
    today: {
      card: "bg-gradient-to-br from-amber-500/10 to-orange-500/10 border-amber-500/30",
      icon: "bg-amber-500/20 text-amber-600",
      text: "text-amber-600",
    },
    urgent: {
      card: "bg-gradient-to-br from-amber-500/10 to-yellow-500/10 border-amber-500/20",
      icon: "bg-amber-500/20 text-amber-600",
      text: "text-amber-600",
    },
    normal: {
      card: "bg-gradient-to-br from-violet-500/10 to-indigo-500/10 border-violet-500/20",
      icon: "bg-violet-500/20 text-violet-600",
      text: "text-violet-600",
    },
  };

  const currentStyle = styles[variant];

  const getDisplayText = () => {
    if (isOverdue) return `${Math.abs(days)} day${Math.abs(days) !== 1 ? "s" : ""} ago`;
    if (isToday) return "Today!";
    return `${days} day${days !== 1 ? "s" : ""}`;
  };

  const getIcon = () => {
    if (isOverdue) return <AlertTriangle className="h-5 w-5" />;
    if (isToday) return <CheckCircle2 className="h-5 w-5" />;
    return <Calendar className="h-5 w-5" />;
  };

  return (
    <Card className={cn("transition-all hover:shadow-md", currentStyle.card)}>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg", currentStyle.icon)}>
            {getIcon()}
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {isOverdue ? "Rehearsal was" : isToday ? "Rehearsal" : "Days until"}
            </p>
            <p className={cn("text-2xl font-bold tabular-nums", currentStyle.text)}>
              {getDisplayText()}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              {cycleLabel}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

