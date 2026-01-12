/**
 * Assignments Page
 *
 * Displays practice assignments with filtering.
 * Uses CONTRACT LOCK protocol - client.ts wrappers only.
 */

import { useState, useEffect } from "react";
import { format, parseISO } from "date-fns";
import {
  listAssignments,
  getActiveCycle,
  type ListAssignmentsFilters,
} from "@/lib/api/client";
import type {
  AssignmentResponse,
  AssignmentType,
  AssignmentScope,
  Priority,
} from "@/lib/api/types";
import { useAuth } from "@/lib/auth";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { ListSkeleton } from "@/components/ui/loading";
import {
  NoAssignmentsEmptyState,
  NoFilterResultsEmptyState,
  NoCycleEmptyState,
} from "@/components/ui/empty-state";
import { toastError } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Music,
  Headphones,
  BookOpen,
  Dumbbell,
  Calendar,
  Filter,
  Globe,
  Users,
  X,
} from "lucide-react";

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

const typeConfig: Record<AssignmentType, { label: string; icon: React.ReactNode }> = {
  SONG_WORK: { label: "Song Work", icon: <Music className="h-4 w-4" /> },
  TECHNIQUE: { label: "Technique", icon: <Dumbbell className="h-4 w-4" /> },
  MEMORIZATION: { label: "Memorization", icon: <BookOpen className="h-4 w-4" /> },
  LISTENING: { label: "Listening", icon: <Headphones className="h-4 w-4" /> },
};

const scopeConfig: Record<AssignmentScope, { label: string; icon: React.ReactNode }> = {
  TEAM: { label: "Team", icon: <Globe className="h-3 w-3" /> },
  SECTION: { label: "Section", icon: <Users className="h-3 w-3" /> },
};

export function AssignmentsPage() {
  const { primaryTeam } = useAuth();
  const [assignments, setAssignments] = useState<AssignmentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [cycleId, setCycleId] = useState<string | null>(null);

  // Filters
  const [filters, setFilters] = useState<ListAssignmentsFilters>({});

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

  // Fetch assignments
  useEffect(() => {
    if (!cycleId) {
      setLoading(false);
      return;
    }

    const fetchAssignments = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await listAssignments(cycleId, filters);
        setAssignments(response.items);
        setNextCursor(response.next_cursor);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load assignments");
      } finally {
        setLoading(false);
      }
    };

    fetchAssignments();
  }, [cycleId, filters]);

  const handleLoadMore = async () => {
    if (!cycleId || !nextCursor) return;

    try {
      const response = await listAssignments(cycleId, filters, nextCursor);
      setAssignments((prev) => [...prev, ...response.items]);
      setNextCursor(response.next_cursor);
    } catch (err) {
      toastError(err instanceof Error ? err.message : "Failed to load more assignments");
    }
  };

  const handleClearFilters = () => {
    setFilters({});
  };

  const hasActiveFilters = Object.keys(filters).length > 0;

  const AssignmentCard = ({ assignment }: { assignment: AssignmentResponse }) => {
    const priority = priorityConfig[assignment.priority];
    const type = typeConfig[assignment.type];
    const scope = scopeConfig[assignment.scope];

    return (
      <Card className="hover:shadow-md transition-shadow">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            {/* Type icon */}
            <div
              className={cn(
                "p-2 rounded-lg",
                assignment.priority === "BLOCKING" && "bg-red-500/10",
                assignment.priority === "MEDIUM" && "bg-amber-500/10",
                assignment.priority === "LOW" && "bg-slate-500/10"
              )}
            >
              {type.icon}
            </div>

            <div className="flex-1 min-w-0">
              {/* Header */}
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate">{assignment.title}</h3>
                  {assignment.notes && (
                    <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
                      {assignment.notes}
                    </p>
                  )}
                </div>
              </div>

              {/* Badges */}
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <Badge variant="outline" className={cn("text-xs py-0 h-5", priority.className)}>
                  {priority.label}
                </Badge>
                <Badge variant="outline" className="text-xs py-0 h-5">
                  {type.label}
                </Badge>
                <Badge
                  variant="secondary"
                  className="text-xs py-0 h-5 flex items-center gap-1"
                >
                  {scope.icon}
                  {scope.label}
                  {assignment.section && `: ${assignment.section}`}
                </Badge>
                {assignment.song_ref && (
                  <Badge variant="outline" className="text-xs py-0 h-5">
                    🎵 {assignment.song_ref}
                  </Badge>
                )}
              </div>

              {/* Footer */}
              {assignment.due_at && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Calendar className="h-3 w-3" />
                  <span>Due {format(parseISO(assignment.due_at), "MMM d, yyyy")}</span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const FilterBar = () => (
    <div className="flex flex-wrap items-center gap-2">
      <Select
        value={filters.type || "ALL"}
        onValueChange={(value) =>
          setFilters((prev) => ({
            ...prev,
            type: value !== "ALL" ? value : undefined,
          }))
        }
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Type" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="ALL">All Types</SelectItem>
          <SelectItem value="SONG_WORK">Song Work</SelectItem>
          <SelectItem value="TECHNIQUE">Technique</SelectItem>
          <SelectItem value="MEMORIZATION">Memorization</SelectItem>
          <SelectItem value="LISTENING">Listening</SelectItem>
        </SelectContent>
      </Select>

      <Select
        value={filters.priority || "ALL"}
        onValueChange={(value) =>
          setFilters((prev) => ({
            ...prev,
            priority: value !== "ALL" ? value : undefined,
          }))
        }
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Priority" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="ALL">All Priorities</SelectItem>
          <SelectItem value="BLOCKING">Blocking</SelectItem>
          <SelectItem value="MEDIUM">Medium</SelectItem>
          <SelectItem value="LOW">Low</SelectItem>
        </SelectContent>
      </Select>

      <Select
        value={filters.scope || "ALL"}
        onValueChange={(value) =>
          setFilters((prev) => ({
            ...prev,
            scope: value !== "ALL" ? value : undefined,
          }))
        }
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Scope" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="ALL">All Scopes</SelectItem>
          <SelectItem value="TEAM">Team</SelectItem>
          <SelectItem value="SECTION">Section</SelectItem>
        </SelectContent>
      </Select>

      {hasActiveFilters && (
        <Button variant="ghost" size="sm" onClick={handleClearFilters}>
          <X className="h-4 w-4 mr-1" />
          Clear filters
        </Button>
      )}
    </div>
  );

  if (!cycleId && !loading) {
    return (
      <div className="min-h-full bg-neutral-50">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Assignments</h1>
              <p className="text-muted-foreground">Practice assignments for your team</p>
            </div>
            <NoCycleEmptyState />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-neutral-50">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Assignments</h1>
            <p className="text-muted-foreground">Practice assignments for your team</p>
          </div>

          {/* Filters - Desktop */}
          <div className="hidden md:block">
            <FilterBar />
          </div>

          {/* Filters - Mobile */}
          <div className="md:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" className="w-full">
                  <Filter className="h-4 w-4 mr-2" />
                  Filters
                  {hasActiveFilters && (
                    <Badge variant="secondary" className="ml-2">
                      {Object.keys(filters).length}
                    </Badge>
                  )}
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetHeader>
                  <SheetTitle>Filter Assignments</SheetTitle>
                </SheetHeader>
                <div className="mt-4 space-y-4">
                  <FilterBar />
                </div>
              </SheetContent>
            </Sheet>
          </div>

          {/* Content */}
          {loading ? (
            <ListSkeleton count={6} variant="card" />
          ) : error ? (
            <Card>
              <CardContent className="p-6">
                <p className="text-center text-red-600">{error}</p>
              </CardContent>
            </Card>
          ) : assignments.length === 0 ? (
            hasActiveFilters ? (
              <NoFilterResultsEmptyState onAction={handleClearFilters} />
            ) : (
              <NoAssignmentsEmptyState />
            )
          ) : (
            <>
              {/* Assignments grid */}
              <div className="grid gap-4 md:grid-cols-2">
                {assignments.map((assignment) => (
                  <AssignmentCard key={assignment.id} assignment={assignment} />
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
