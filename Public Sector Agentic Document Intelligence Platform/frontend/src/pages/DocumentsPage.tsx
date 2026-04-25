import { ChangeEvent, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { FileUp, FileText, Hash, Layers } from 'lucide-react';
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
} from '@/api/documents';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { useToast } from '@/components/ui/Toast';
import type { DocumentListItem, DocumentStatus } from '@/api/schemas';
import { useAuth } from '@/state/auth';

export function DocumentsPage() {
  const { user } = useAuth();
  const fileInput = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const { push } = useToast();
  const [search, setSearch] = useState('');

  const docs = useQuery({
    queryKey: ['documents'],
    queryFn: () => listDocuments(1, 100),
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return false;
      const inflight = data.items.some(
        (d) => d.status !== 'ready' && d.status !== 'failed',
      );
      return inflight ? 4000 : false;
    },
  });

  const upload = useMutation({
    mutationFn: (file: File) => uploadDocument(file),
    onSuccess: (resp) => {
      void qc.invalidateQueries({ queryKey: ['documents'] });
      push(
        resp.duplicate
          ? 'Document already on file — listing refreshed.'
          : 'Upload accepted. Ingestion underway.',
        resp.duplicate ? 'info' : 'success',
      );
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : 'Upload failed';
      push(message, 'error');
    },
  });

  const remove = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['documents'] });
      push('Document withdrawn.', 'info');
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : 'Withdraw failed';
      push(message, 'error');
    },
  });

  function pick() {
    fileInput.current?.click();
  }

  function onFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) upload.mutate(file);
    e.target.value = '';
  }

  const items = docs.data?.items ?? [];
  const filtered = items.filter((d) =>
    d.filename.toLowerCase().includes(search.toLowerCase()),
  );
  const ready = items.filter((d) => d.status === 'ready').length;
  const inflight = items.filter(
    (d) => d.status !== 'ready' && d.status !== 'failed',
  ).length;
  const failed = items.filter((d) => d.status === 'failed').length;

  return (
    <div className="stagger">
      <header className="mb-10">
        <p className="rubric">002 — The Reading Room</p>
        <h1 className="display text-5xl mt-2">Documents on file.</h1>
        <p className="mt-4 text-base text-ink-80 max-w-prose leading-relaxed">
          Every document admitted here is parsed, chunked, embedded, and made
          retrievable. Ingestion runs asynchronously; this listing
          self-refreshes while work is in progress.
        </p>
      </header>

      <div className="flex flex-wrap items-end justify-between gap-6 mb-10">
        <div className="flex gap-10 flex-wrap">
          <Stat rubric="ready" value={ready} icon={<FileText size={14} />} />
          <Stat
            rubric="in flight"
            value={inflight}
            tone={inflight > 0 ? 'leaf' : 'neutral'}
            icon={<Layers size={14} />}
          />
          <Stat
            rubric="failed"
            value={failed}
            tone={failed > 0 ? 'seal' : 'neutral'}
            icon={<Hash size={14} />}
          />
        </div>

        <div className="flex items-end gap-4">
          <label className="block w-72">
            <span className="rubric">filter</span>
            <input
              type="search"
              className="field"
              placeholder="filename contains…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
            onChange={onFile}
            className="hidden"
          />
          <Button
            onClick={pick}
            loading={upload.isPending}
            leftIcon={<FileUp size={14} />}
          >
            Submit a PDF
          </Button>
        </div>
      </div>

      <hr className="rule-double mb-10" />

      {docs.isLoading ? (
        <Card rubric="loading" title="Reading the registry…">
          <Skeleton rows={6} />
        </Card>
      ) : docs.isError ? (
        <ErrorState
          title="Could not load documents"
          description={
            docs.error instanceof Error ? docs.error.message : 'Unknown error'
          }
          action={<Button onClick={() => docs.refetch()}>Retry</Button>}
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          rubric="empty registry"
          title={items.length === 0 ? 'No documents on file.' : 'No matches.'}
          description={
            items.length === 0
              ? `Submit a PDF to begin. The seeded corpus is ingested on first boot — if this list is empty, ${user?.email ?? 'this analyst'} has not uploaded anything yet.`
              : 'Adjust your filter to widen the search.'
          }
          action={
            items.length === 0 ? (
              <Button onClick={pick} leftIcon={<FileUp size={14} />}>
                Submit a PDF
              </Button>
            ) : null
          }
        />
      ) : (
        <ul className="border-t border-hair border-rule">
          {filtered.map((doc) => (
            <DocumentRow
              key={doc.id}
              doc={doc}
              onDelete={() => remove.mutate(doc.id)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function Stat({
  rubric,
  value,
  icon,
  tone = 'neutral',
}: {
  rubric: string;
  value: number;
  icon: React.ReactNode;
  tone?: 'neutral' | 'seal' | 'leaf';
}) {
  const colorClass =
    tone === 'seal' ? 'text-seal' : tone === 'leaf' ? 'text-leaf-deep' : 'text-ink';
  return (
    <div className="flex items-baseline gap-3">
      <span className="text-ink-60">{icon}</span>
      <div>
        <p className="rubric">{rubric}</p>
        <p className={`display text-3xl datum ${colorClass}`}>{value}</p>
      </div>
    </div>
  );
}

const STATUS_TONE: Record<DocumentStatus, 'neutral' | 'seal' | 'forest' | 'leaf'> = {
  pending: 'neutral',
  extracting: 'leaf',
  chunking: 'leaf',
  embedding: 'leaf',
  ready: 'forest',
  failed: 'seal',
};

function DocumentRow({
  doc,
  onDelete,
}: {
  doc: DocumentListItem;
  onDelete: () => void;
}) {
  const created = new Date(doc.created_at).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <li className="grid grid-cols-[7rem_1fr_auto] items-baseline gap-6 border-b border-hair border-rule-soft py-5">
      <span className="datum text-2xs text-ink-60 uppercase tracking-rubric">
        {created}
      </span>
      <div className="min-w-0">
        <p className="display text-xl truncate">{doc.filename}</p>
        <div className="mt-1.5 flex items-center gap-3 text-2xs datum text-ink-60 uppercase tracking-[0.06em]">
          <span>{doc.page_count.toLocaleString()} pp</span>
          <span aria-hidden>·</span>
          <span>{(doc.byte_size / 1024).toLocaleString(undefined, { maximumFractionDigits: 0 })} KB</span>
          <span aria-hidden>·</span>
          <span>{doc.chunk_count.toLocaleString()} chunks</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Badge tone={STATUS_TONE[doc.status]}>{doc.status}</Badge>
        <button onClick={onDelete} className="btn-ghost" aria-label={`Withdraw ${doc.filename}`}>
          Withdraw
        </button>
      </div>
    </li>
  );
}
