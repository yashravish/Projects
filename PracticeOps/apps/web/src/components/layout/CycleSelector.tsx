/**
 * Cycle Selector Component
 *
 * Displays the active cycle with date, label, and countdown.
 * Uses Popover for future cycle history/selection.
 */

import { useQuery } from "@tanstack/react-query";
import { Calendar, ChevronDown, Clock } from "lucide-react";
import { differenceInDays, format, isToday, isPast, parseISO } from "date-fns";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { useTeam } from "@/lib/auth";
import { api } from "@/lib/api/client";

export function CycleSelector() {
  const { team } = useTeam();

  const { data, isLoading, error } = useQuery({
    queryKey: ["activeCycle", team?.team_id],
    queryFn: () => api.getActiveCycle(team!.team_id),
    enabled: !!team?.team_id,
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2">
        <Skeleton className="h-8 w-32" />
      </div>
    );
  }

  if (error || !data?.cycle) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Calendar className="h-4 w-4" />
        <span className="text-sm">No active cycle</span>
      </div>
    );
  }

  const cycle = data.cycle;
  const cycleDate = parseISO(cycle.date);
  const countdown = getCountdown(cycleDate);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="h-auto gap-2 px-3 py-2"
          aria-label="Select cycle"
        >
          <Calendar className="h-4 w-4 text-primary" />
          <div className="flex flex-col items-start">
            <span className="text-sm font-medium">{cycle.name}</span>
            <span className="text-xs text-muted-foreground">
              {format(cycleDate, "MMM d, yyyy")}
            </span>
          </div>
          <Badge
            variant={countdown.variant}
            className="ml-1 text-xs"
          >
            {countdown.label}
          </Badge>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72" align="end">
        <div className="space-y-4">
          <div className="space-y-1">
            <h4 className="font-medium">{cycle.name}</h4>
            <p className="text-sm text-muted-foreground">
              {format(cycleDate, "EEEE, MMMM d, yyyy")}
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg bg-muted/50 p-3">
            <Clock className="h-5 w-5 text-primary" />
            <div>
              <p className="text-sm font-medium">{countdown.label}</p>
              <p className="text-xs text-muted-foreground">
                {countdown.description}
              </p>
            </div>
          </div>

          {/* Future: Cycle history/selection */}
          <p className="text-xs text-muted-foreground">
            Showing the current active rehearsal cycle
          </p>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// Countdown calculation
function getCountdown(date: Date): {
  label: string;
  description: string;
  variant: "default" | "secondary" | "destructive" | "outline";
} {
  if (isToday(date)) {
    return {
      label: "Today",
      description: "Rehearsal is today!",
      variant: "default",
    };
  }

  const days = differenceInDays(date, new Date());

  if (isPast(date)) {
    const daysAgo = Math.abs(days);
    return {
      label: `${daysAgo}d ago`,
      description: `Rehearsal was ${daysAgo} day${daysAgo !== 1 ? "s" : ""} ago`,
      variant: "secondary",
    };
  }

  if (days === 1) {
    return {
      label: "Tomorrow",
      description: "Rehearsal is tomorrow",
      variant: "default",
    };
  }

  if (days <= 3) {
    return {
      label: `${days}d`,
      description: `Rehearsal in ${days} days`,
      variant: "destructive",
    };
  }

  return {
    label: `${days}d`,
    description: `Rehearsal in ${days} days`,
    variant: "outline",
  };
}

