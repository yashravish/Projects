/**
 * Stats Card Component - Display a single stat with icon
 */

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  variant?: "default" | "accent" | "warning" | "success";
}

export function StatsCard({
  title,
  value,
  subtitle,
  icon,
  variant = "default",
}: StatsCardProps) {
  const variantStyles = {
    default: "bg-card",
    accent: "bg-gradient-to-br from-violet-500/10 to-fuchsia-500/10 border-violet-500/20",
    warning: "bg-gradient-to-br from-amber-500/10 to-orange-500/10 border-amber-500/20",
    success: "bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border-emerald-500/20",
  };

  return (
    <Card className={cn("transition-all hover:shadow-md", variantStyles[variant])}>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            {icon}
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {title}
            </p>
            <p className="text-2xl font-bold tabular-nums">{value}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground truncate">{subtitle}</p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

