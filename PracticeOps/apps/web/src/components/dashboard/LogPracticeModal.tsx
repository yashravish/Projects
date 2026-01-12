/**
 * Log Practice Modal - Fast path for logging practice sessions
 * Target: 60 seconds from open to submit
 */

import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ButtonSpinner } from "@/components/ui/loading";
import { toastSuccess, toastError } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { 
  Minus, 
  Plus, 
  AlertTriangle, 
  Ticket, 
  Star,
  Check,
} from "lucide-react";
import { createPracticeLog, ApiClientError } from "@/lib/api/client";
import type { 
  AssignmentSummary, 
  SuggestedTicket,
  Priority,
} from "@/lib/api/types";

interface LogPracticeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  cycleId: string | null;
  assignments: AssignmentSummary[];
  defaultDuration: number;
  teamId: string;
  isMobile?: boolean;
}

const priorityConfig: Record<Priority, { label: string; className: string; order: number }> = {
  BLOCKING: {
    label: "Blocking",
    className: "bg-red-500/15 text-red-600 border-red-500/30 hover:bg-red-500/25",
    order: 0,
  },
  MEDIUM: {
    label: "Medium",
    className: "bg-amber-500/15 text-amber-600 border-amber-500/30 hover:bg-amber-500/25",
    order: 1,
  },
  LOW: {
    label: "Low",
    className: "bg-slate-500/15 text-slate-600 border-slate-500/30 hover:bg-slate-500/25",
    order: 2,
  },
};

export function LogPracticeModal({
  open,
  onOpenChange,
  cycleId,
  assignments,
  defaultDuration,
  teamId,
  isMobile = false,
}: LogPracticeModalProps) {
  const queryClient = useQueryClient();

  // Form state
  const [duration, setDuration] = useState(defaultDuration);
  const [selectedAssignments, setSelectedAssignments] = useState<string[]>([]);
  const [blocked, setBlocked] = useState(false);
  const [notes, setNotes] = useState("");
  const [rating, setRating] = useState<number | null>(null);

  // Post-submit state for blocked flow
  const [suggestedTicket, setSuggestedTicket] = useState<SuggestedTicket | null>(null);
  const [showTicketPrompt, setShowTicketPrompt] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Reset form
  const resetForm = useCallback(() => {
    setDuration(defaultDuration);
    setSelectedAssignments([]);
    setBlocked(false);
    setNotes("");
    setRating(null);
    setSuggestedTicket(null);
    setShowTicketPrompt(false);
    setError(null);
  }, [defaultDuration]);

  // Handle close
  const handleClose = useCallback(() => {
    resetForm();
    onOpenChange(false);
  }, [resetForm, onOpenChange]);

  // Create practice log mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      if (!cycleId) throw new Error("No active cycle");
      
      return createPracticeLog(cycleId, {
        duration_min: duration,
        assignment_ids: selectedAssignments.length > 0 ? selectedAssignments : undefined,
        blocked_flag: blocked,
        notes: notes.trim() || undefined,
        rating_1_5: rating || undefined,
      });
    },
    onSuccess: (data) => {
      // Invalidate dashboard query to refresh data
      queryClient.invalidateQueries({ queryKey: ["dashboard", teamId] });

      // Check for suggested ticket (blocked flow)
      if (data.suggested_ticket) {
        setSuggestedTicket(data.suggested_ticket);
        setShowTicketPrompt(true);
      } else {
        toastSuccess("Practice logged!");
        handleClose();
      }
    },
    onError: (err) => {
      if (err instanceof ApiClientError) {
        if (err.code === "UNAUTHORIZED") {
          setError("Session expired. Please log in again.");
          toastError("Session expired. Please log in again.");
        } else {
          setError(err.message);
          toastError(err.message);
        }
      } else {
        setError("Failed to log practice. Please try again.");
        toastError("Failed to log practice. Please try again.");
      }
    },
  });

  // Handle submit
  const handleSubmit = () => {
    setError(null);
    createMutation.mutate();
  };

  // Duration controls
  const adjustDuration = (delta: number) => {
    setDuration((prev) => Math.max(1, Math.min(600, prev + delta)));
  };

  // Toggle assignment selection
  const toggleAssignment = (id: string) => {
    setSelectedAssignments((prev) =>
      prev.includes(id) ? prev.filter((a) => a !== id) : [...prev, id]
    );
  };

  // Group assignments by priority
  const groupedAssignments = assignments.reduce(
    (acc, assignment) => {
      acc[assignment.priority] = acc[assignment.priority] || [];
      acc[assignment.priority].push(assignment);
      return acc;
    },
    {} as Record<Priority, AssignmentSummary[]>
  );

  // Sort priority groups
  const sortedPriorities = (Object.keys(groupedAssignments) as Priority[]).sort(
    (a, b) => priorityConfig[a].order - priorityConfig[b].order
  );

  // Main form content
  const FormContent = (
    <div className="space-y-6">
      {/* Error display */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Duration input */}
      <div className="space-y-2">
        <Label htmlFor="duration" className="text-sm font-medium">
          Duration (minutes)
        </Label>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => adjustDuration(-5)}
            disabled={duration <= 5}
            className="h-10 w-10"
          >
            <Minus className="h-4 w-4" />
          </Button>
          <Input
            id="duration"
            type="number"
            value={duration}
            onChange={(e) => setDuration(Math.max(1, Math.min(600, parseInt(e.target.value) || 0)))}
            className="w-20 text-center text-lg font-semibold"
            min={1}
            max={600}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => adjustDuration(5)}
            disabled={duration >= 600}
            className="h-10 w-10"
          >
            <Plus className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground ml-2">min</span>
        </div>
      </div>

      {/* Assignments selection */}
      {assignments.length > 0 && (
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            What did you work on?{" "}
            <span className="text-muted-foreground font-normal">(optional)</span>
          </Label>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {sortedPriorities.map((priority) => (
              <div key={priority} className="space-y-1.5">
                <p className="text-xs text-muted-foreground uppercase tracking-wide">
                  {priorityConfig[priority].label}
                </p>
                <div className="flex flex-wrap gap-2">
                  {groupedAssignments[priority].map((assignment) => {
                    const isSelected = selectedAssignments.includes(assignment.id);
                    return (
                      <button
                        key={assignment.id}
                        type="button"
                        onClick={() => toggleAssignment(assignment.id)}
                        className={cn(
                          "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm border transition-all",
                          isSelected
                            ? "bg-primary text-primary-foreground border-primary"
                            : priorityConfig[priority].className
                        )}
                      >
                        {isSelected && <Check className="h-3 w-3" />}
                        <span className="truncate max-w-[200px]">{assignment.title}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Blocked toggle */}
      <div className="flex items-center justify-between rounded-lg border p-3 bg-muted/30">
        <div className="space-y-0.5">
          <Label htmlFor="blocked" className="text-sm font-medium cursor-pointer">
            I got blocked
          </Label>
          <p className="text-xs text-muted-foreground">
            Had trouble with something? We'll help you create a ticket.
          </p>
        </div>
        <Switch
          id="blocked"
          checked={blocked}
          onCheckedChange={setBlocked}
        />
      </div>

      {/* Optional fields (collapsed by default) */}
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="optional" className="border-none">
          <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:text-foreground">
            Optional details
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-2">
            {/* Notes */}
            <div className="space-y-2">
              <Label htmlFor="notes" className="text-sm font-medium">
                Notes
              </Label>
              <textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="What did you focus on? Any breakthroughs?"
                className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
                maxLength={2000}
              />
              <p className="text-xs text-muted-foreground text-right">
                {notes.length}/2000
              </p>
            </div>

            {/* Rating */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                How was your practice? (1-5)
              </Label>
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    onClick={() => setRating(rating === star ? null : star)}
                    className={cn(
                      "p-1 rounded transition-colors",
                      rating && star <= rating
                        ? "text-amber-500"
                        : "text-muted-foreground hover:text-amber-400"
                    )}
                  >
                    <Star
                      className={cn(
                        "h-6 w-6",
                        rating && star <= rating && "fill-current"
                      )}
                    />
                  </button>
                ))}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {/* Submit button */}
      <Button
        onClick={handleSubmit}
        disabled={createMutation.isPending || !cycleId}
        className="w-full"
        size="lg"
      >
        {createMutation.isPending ? (
          <>
            <ButtonSpinner className="mr-2" />
            Logging...
          </>
        ) : (
          "Log Practice"
        )}
      </Button>

      {!cycleId && (
        <p className="text-xs text-center text-muted-foreground">
          No active cycle. Practice logging is disabled.
        </p>
      )}
    </div>
  );

  // Ticket suggestion prompt
  const TicketPromptContent = suggestedTicket && (
    <div className="space-y-4">
      <Alert className="border-amber-500/30 bg-amber-500/10">
        <Ticket className="h-4 w-4 text-amber-600" />
        <AlertTitle className="text-amber-600">Practice logged!</AlertTitle>
        <AlertDescription>
          You mentioned being blocked. Would you like to create a ticket to track this?
        </AlertDescription>
      </Alert>

      <div className="rounded-lg border p-4 space-y-3 bg-muted/30">
        <p className="text-sm font-medium">Suggested ticket details:</p>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Title:</span>
            <span className="font-medium">{suggestedTicket.title_suggestion}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Due:</span>
            <span>{suggestedTicket.due_date}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">Priority:</span>
            <Badge variant="outline" className={priorityConfig[suggestedTicket.priority_default].className}>
              {priorityConfig[suggestedTicket.priority_default].label}
            </Badge>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Visibility:</span>
            <span className="capitalize">{suggestedTicket.visibility_default.toLowerCase()}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <Button variant="outline" onClick={handleClose} className="flex-1">
          Skip
        </Button>
        <Button
          onClick={() => {
            // TODO: Navigate to ticket creation with pre-filled data
            // For now, just close
            handleClose();
          }}
          className="flex-1"
        >
          <Ticket className="mr-2 h-4 w-4" />
          Create Ticket
        </Button>
      </div>
    </div>
  );

  // Render based on device
  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={handleClose}>
        <SheetContent side="bottom" className="h-[85vh]">
          <SheetHeader>
            <SheetTitle>
              {showTicketPrompt ? "Create a Ticket?" : "Log Practice"}
            </SheetTitle>
            <SheetDescription>
              {showTicketPrompt
                ? "Track your blocker to get help"
                : "Record your practice session"}
            </SheetDescription>
          </SheetHeader>
          <div className="mt-4 overflow-y-auto pb-safe">
            {showTicketPrompt ? TicketPromptContent : FormContent}
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>
            {showTicketPrompt ? "Create a Ticket?" : "Log Practice"}
          </DialogTitle>
          <DialogDescription>
            {showTicketPrompt
              ? "Track your blocker to get help"
              : "Record your practice session"}
          </DialogDescription>
        </DialogHeader>
        {showTicketPrompt ? TicketPromptContent : FormContent}
      </DialogContent>
    </Dialog>
  );
}

