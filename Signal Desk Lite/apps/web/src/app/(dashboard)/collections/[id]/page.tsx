'use client';

import { useState, useRef } from 'react';
import { useParams } from 'next/navigation';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { apiRequest, uploadFile } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { FileText, Upload, Send, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'sonner';

interface Document {
  id: string;
  originalName: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  chunkCount: number;
  createdAt: string;
}

interface Citation {
  chunkId: string;
  documentName: string;
  content: string;
  relevanceScore: number;
}

interface QueryResult {
  answer: string;
  citations: Citation[];
  keyFacts: Array<{ fact: string; source: string }>;
  summary: string;
}

export default function CollectionDetailPage() {
  const params = useParams();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [question, setQuestion] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set());

  const { data: collection } = useQuery({
    queryKey: ['collection', params.id],
    queryFn: () => apiRequest(`/v1/collections/${params.id}`),
  });

  const { data: documents, isLoading: documentsLoading } = useQuery({
    queryKey: ['documents', params.id],
    queryFn: () => apiRequest<{ documents: Document[] }>(`/v1/collections/${params.id}/documents`),
    refetchInterval: 5000,
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadFile(`/v1/collections/${params.id}/documents`, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', params.id] });
      toast.success('Document uploaded successfully');
    },
    onError: () => {
      toast.error('Failed to upload document');
    },
  });

  const queryMutation = useMutation({
    mutationFn: (q: string) =>
      apiRequest<QueryResult>(`/v1/collections/${params.id}/query`, {
        method: 'POST',
        body: JSON.stringify({ question: q }),
      }),
    onSuccess: (result) => {
      if (result.data) {
        setQueryResult(result.data);
      }
    },
    onError: () => {
      toast.error('Failed to query documents');
    },
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadMutation.mutate(file);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleQuery = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim()) {
      queryMutation.mutate(question);
    }
  };

  const toggleCitation = (chunkId: string) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) {
        next.delete(chunkId);
      } else {
        next.add(chunkId);
      }
      return next;
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'default';
      case 'PROCESSING':
        return 'secondary';
      case 'FAILED':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-4xl font-display font-bold">
          {collection?.data?.collection?.name || 'Collection'}
        </h1>
        {collection?.data?.collection?.description && (
          <p className="text-muted-foreground mt-2">{collection.data.collection.description}</p>
        )}
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-display font-semibold">Documents</h2>
            <Button onClick={() => fileInputRef.current?.click()} disabled={uploadMutation.isPending}>
              <Upload className="h-4 w-4 mr-2" />
              Upload
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.txt,.md"
              onChange={handleFileUpload}
            />
          </div>

          {documentsLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          ) : documents?.data?.documents.length === 0 ? (
            <Card className="glass">
              <CardContent className="py-12 text-center">
                <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-semibold mb-2">No documents yet</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Upload your first document to get started
                </p>
                <Button onClick={() => fileInputRef.current?.click()}>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Document
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-2">
              {documents?.data?.documents.map((doc) => (
                <Card key={doc.id} className="glass">
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1">
                      <FileText className="h-5 w-5 text-primary" />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{doc.originalName}</p>
                        <p className="text-sm text-muted-foreground">
                          {doc.chunkCount} chunks
                        </p>
                      </div>
                    </div>
                    <Badge variant={getStatusColor(doc.status)}>{doc.status}</Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-display font-semibold">Query</h2>

          <form onSubmit={handleQuery}>
            <div className="flex gap-2">
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about your documents..."
                className="flex-1"
              />
              <Button type="submit" disabled={queryMutation.isPending || !question.trim()}>
                {queryMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </form>

          {queryResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <Card className="glass">
                <CardHeader>
                  <CardTitle>Answer</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="whitespace-pre-wrap">{queryResult.answer}</p>
                </CardContent>
              </Card>

              {queryResult.summary && (
                <Card className="glass">
                  <CardHeader>
                    <CardTitle>Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm">{queryResult.summary}</p>
                  </CardContent>
                </Card>
              )}

              {queryResult.keyFacts && queryResult.keyFacts.length > 0 && (
                <Card className="glass">
                  <CardHeader>
                    <CardTitle>Key Facts</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {queryResult.keyFacts.map((fact, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-primary mt-1">•</span>
                          <span className="text-sm">{fact.fact}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {queryResult.citations && queryResult.citations.length > 0 && (
                <Card className="glass">
                  <CardHeader>
                    <CardTitle>Citations</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {queryResult.citations.map((citation, index) => (
                      <div key={citation.chunkId} className="border border-border rounded-lg">
                        <button
                          onClick={() => toggleCitation(citation.chunkId)}
                          className="w-full p-3 flex items-center justify-between hover:bg-accent/50 rounded-lg transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">[{index + 1}]</Badge>
                            <span className="text-sm font-medium">{citation.documentName}</span>
                            <span className="text-xs text-muted-foreground">
                              Score: {(citation.relevanceScore * 100).toFixed(1)}%
                            </span>
                          </div>
                          {expandedCitations.has(citation.chunkId) ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </button>
                        {expandedCitations.has(citation.chunkId) && (
                          <div className="p-3 border-t border-border">
                            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                              {citation.content}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
