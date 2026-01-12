/**
 * Streak Card - Shows practice streak with celebration
 */

import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { Flame, Sparkles, Trophy, Star } from "lucide-react";
import { useEffect, useState } from "react";

interface StreakCardProps {
  streakDays: number;
  practiceDays: number;
  totalSessions: number;
}

export function StreakCard({ streakDays, practiceDays, totalSessions }: StreakCardProps) {
  const [showCelebration, setShowCelebration] = useState(false);

  // Check for milestone celebration
  const milestones = [7, 14, 21, 30, 60, 90];
  const isMilestone = milestones.includes(streakDays);
  const nextMilestone = milestones.find((m) => m > streakDays) || streakDays + 7;
  const progressToNext = (streakDays / nextMilestone) * 100;

  useEffect(() => {
    if (isMilestone && streakDays > 0) {
      setShowCelebration(true);
      const timer = setTimeout(() => setShowCelebration(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [isMilestone, streakDays]);

  const getStreakColor = () => {
    if (streakDays >= 14) return "text-orange-500";
    if (streakDays >= 7) return "text-amber-500";
    if (streakDays > 0) return "text-yellow-500";
    return "text-muted-foreground";
  };

  const getStreakBg = () => {
    if (streakDays >= 14) return "bg-gradient-to-br from-orange-500/15 to-red-500/15 border-orange-500/30";
    if (streakDays >= 7) return "bg-gradient-to-br from-amber-500/15 to-orange-500/15 border-amber-500/30";
    if (streakDays > 0) return "bg-gradient-to-br from-yellow-500/10 to-amber-500/10 border-yellow-500/20";
    return "bg-muted/50";
  };

  const getStreakMessage = () => {
    if (streakDays === 0) return "Start your streak today!";
    if (streakDays >= 30) return "Legendary consistency!";
    if (streakDays >= 14) return "On fire! 🔥";
    if (streakDays >= 7) return "One week strong!";
    if (streakDays >= 3) return "Building momentum!";
    return "Keep it going!";
  };

  return (
    <Card className={cn("relative overflow-hidden transition-all hover:shadow-md", getStreakBg())}>
      {/* Celebration overlay */}
      {showCelebration && (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-amber-500/20 to-orange-500/20 backdrop-blur-sm z-10 animate-in fade-in duration-300">
          <div className="text-center">
            <Trophy className="h-12 w-12 text-amber-500 mx-auto mb-2 animate-bounce" />
            <p className="text-lg font-bold text-amber-600">{streakDays} Day Milestone!</p>
            <div className="flex justify-center gap-1 mt-1">
              <Sparkles className="h-4 w-4 text-amber-400" />
              <Sparkles className="h-4 w-4 text-orange-400" />
              <Sparkles className="h-4 w-4 text-amber-400" />
            </div>
          </div>
        </div>
      )}

      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className={cn(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
            streakDays > 0 ? "bg-amber-500/20" : "bg-muted"
          )}>
            {streakDays >= 7 ? (
              <Star className={cn("h-5 w-5", getStreakColor())} />
            ) : (
              <Flame className={cn("h-5 w-5", getStreakColor())} />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Practice Streak
            </p>
            <div className="flex items-baseline gap-2">
              <p className={cn("text-2xl font-bold tabular-nums", getStreakColor())}>
                {streakDays}
              </p>
              <span className="text-sm text-muted-foreground">days</span>
            </div>
            <p className="text-xs text-muted-foreground">{getStreakMessage()}</p>
          </div>
        </div>

        {/* Progress to next milestone */}
        {streakDays > 0 && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>{practiceDays}/7 days this week</span>
              <span>{totalSessions} sessions</span>
            </div>
            <Progress value={progressToNext} className="h-1.5" />
            <p className="text-xs text-muted-foreground mt-1 text-right">
              {nextMilestone - streakDays} to next milestone
            </p>
          </div>
        )}

        {/* New user encouragement */}
        {streakDays === 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            Log your first practice session to start building your streak!
          </p>
        )}
      </CardContent>
    </Card>
  );
}

