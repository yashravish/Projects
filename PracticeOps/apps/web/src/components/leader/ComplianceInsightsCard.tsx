import {
  BarChart,
  Bar,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { ComplianceInsightsResponse, ComplianceSectionDatum } from "@/lib/api/types";

type ChartDatum = {
  name: string;
  avgDays: number;
  members: number;
  totalDays: number;
};

export function buildComplianceChartData(
  sections: ComplianceSectionDatum[]
): ChartDatum[] {
  return [...sections]
    .sort((a, b) => b.avg_practice_days_7d - a.avg_practice_days_7d)
    .map((section) => ({
      name: section.section,
      avgDays: Number(section.avg_practice_days_7d.toFixed(2)),
      members: section.member_count,
      totalDays: section.total_practice_days_7d,
    }));
}

type ComplianceInsightsCardProps = {
  insights?: ComplianceInsightsResponse;
  isLoading: boolean;
  error?: Error | null;
};

export function ComplianceInsightsCard({
  insights,
  isLoading,
  error,
}: ComplianceInsightsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Practice Compliance by Section</CardTitle>
        <CardDescription>
          Average days logged per member in the last {insights?.window_days ?? 7} days
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-40 w-full" />
          </div>
        )}

        {!isLoading && error && (
          <Alert variant="destructive">
            <AlertDescription>
              {error.message || "Failed to load compliance insights."}
            </AlertDescription>
          </Alert>
        )}

        {!isLoading && !error && (!insights || insights.sections.length === 0) && (
          <p className="text-sm text-muted-foreground text-center py-6">
            No compliance data available yet.
          </p>
        )}

        {!isLoading && !error && insights && insights.sections.length > 0 && (
          <>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={buildComplianceChartData(insights.sections)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip
                    formatter={(value: number, _name, props) => [
                      value,
                      `Avg Days (members: ${props.payload.members})`,
                    ]}
                  />
                  <Bar dataKey="avgDays" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-md border bg-muted/40 px-4 py-3 text-sm">
              <div className="flex items-center gap-2 mb-2">
                <span className="font-medium">Summary</span>
                <Badge variant={insights.summary_source === "openai" ? "default" : "secondary"}>
                  {insights.summary_source === "openai" ? "AI" : "Fallback"}
                </Badge>
              </div>
              <p className="text-muted-foreground">{insights.summary}</p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
