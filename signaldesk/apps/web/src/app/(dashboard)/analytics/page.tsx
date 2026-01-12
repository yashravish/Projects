'use client';

import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { apiRequest } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { FileText, Layers, MessageSquare, DollarSign } from 'lucide-react';

interface Analytics {
  documentCount: number;
  chunkCount: number;
  queryCount: number;
  tokenUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  estimatedCost: {
    embeddingCost: number;
    chatInputCost: number;
    chatOutputCost: number;
    totalCost: number;
  };
}

export default function AnalyticsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => apiRequest<{ analytics: Analytics }>('/v1/analytics'),
  });

  const stats = [
    {
      title: 'Documents',
      value: data?.data?.analytics.documentCount ?? 0,
      icon: FileText,
      color: 'text-blue-500',
    },
    {
      title: 'Chunks',
      value: data?.data?.analytics.chunkCount ?? 0,
      icon: Layers,
      color: 'text-purple-500',
    },
    {
      title: 'Queries',
      value: data?.data?.analytics.queryCount ?? 0,
      icon: MessageSquare,
      color: 'text-green-500',
    },
    {
      title: 'Estimated Cost',
      value: `$${data?.data?.analytics.estimatedCost.totalCost.toFixed(4) ?? '0.0000'}`,
      icon: DollarSign,
      color: 'text-amber-500',
    },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-4xl font-display font-bold">Analytics</h1>
        <p className="text-muted-foreground mt-2">
          Track your usage and estimated costs
        </p>
      </div>

      {isLoading ? (
        <div className="grid md:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      ) : (
        <div className="grid md:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="glass card-hover">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <stat.icon className={`h-8 w-8 ${stat.color}`} />
                  </div>
                  <div className="text-3xl font-display font-bold mb-1">{stat.value}</div>
                  <div className="text-sm text-muted-foreground">{stat.title}</div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {!isLoading && data?.data?.analytics && (
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Token Usage</CardTitle>
              <CardDescription>Breakdown of token consumption</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Prompt Tokens</span>
                <span className="font-semibold">
                  {data.data.analytics.tokenUsage.promptTokens.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Completion Tokens</span>
                <span className="font-semibold">
                  {data.data.analytics.tokenUsage.completionTokens.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <span className="text-sm font-semibold">Total Tokens</span>
                <span className="font-bold text-primary">
                  {data.data.analytics.tokenUsage.totalTokens.toLocaleString()}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="glass">
            <CardHeader>
              <CardTitle>Cost Breakdown</CardTitle>
              <CardDescription>Estimated costs based on OpenAI pricing</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Embeddings</span>
                <span className="font-semibold">
                  ${data.data.analytics.estimatedCost.embeddingCost.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Chat Input</span>
                <span className="font-semibold">
                  ${data.data.analytics.estimatedCost.chatInputCost.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Chat Output</span>
                <span className="font-semibold">
                  ${data.data.analytics.estimatedCost.chatOutputCost.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <span className="text-sm font-semibold">Total Cost</span>
                <span className="font-bold text-amber-500">
                  ${data.data.analytics.estimatedCost.totalCost.toFixed(4)}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
