import { useEffect, useState, type FormEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import type { RequestDetail as RequestDetailType, AISummary } from '../types';
import { getRequest, updateRequest, summarizeRequest } from '../api/client';
import { useAuth } from '../context/AuthContext';
import StatusBadge from '../components/StatusBadge';
import { formatLabel } from '../components/FilterBar';

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function formatShortDate(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

const statuses = [
  { value: 'submitted', label: 'Submitted' },
  { value: 'in_review', label: 'In Review' },
  { value: 'in_progress', label: 'In Progress' },
  { value: 'pending_info', label: 'Pending Info' },
  { value: 'resolved', label: 'Resolved' },
  { value: 'closed', label: 'Closed' },
];

const sectionTitle = 'text-lg font-semibold text-gray-900 mb-4';
const cardClass = 'bg-white rounded-xl shadow-sm border border-gray-200 p-6';
const labelClass = 'text-xs font-medium text-gray-500 uppercase tracking-wide';
const valueClass = 'mt-1 text-sm text-gray-900';

export default function RequestDetail() {
  const { id } = useParams<{ id: string }>();
  const { isManager } = useAuth();
  const [request, setRequest] = useState<RequestDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [summarizing, setSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState('');

  const [mgrStatus, setMgrStatus] = useState('');
  const [mgrOwner, setMgrOwner] = useState('');
  const [mgrNote, setMgrNote] = useState('');
  const [mgrLoading, setMgrLoading] = useState(false);
  const [mgrError, setMgrError] = useState('');

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError('');
    getRequest(Number(id))
      .then((data) => {
        setRequest(data);
        setMgrStatus(data.status);
        setMgrOwner(data.assigned_owner || '');
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load request'))
      .finally(() => setLoading(false));
  }, [id]);

  const handleSummarize = async () => {
    if (!request) return;
    setSummarizing(true);
    setSummaryError('');
    try {
      const summary: AISummary = await summarizeRequest(request.id);
      setRequest({ ...request, ai_summary: summary });
    } catch (err) {
      setSummaryError(err instanceof Error ? err.message : 'Failed to generate summary');
    } finally {
      setSummarizing(false);
    }
  };

  const handleManagerUpdate = async (e: FormEvent) => {
    e.preventDefault();
    if (!request) return;
    setMgrLoading(true);
    setMgrError('');
    try {
      const updated = await updateRequest(request.id, {
        status: mgrStatus,
        assigned_owner: mgrOwner || undefined,
        note: mgrNote || undefined,
      });
      setRequest(updated);
      setMgrNote('');
    } catch (err) {
      setMgrError(err instanceof Error ? err.message : 'Failed to update request');
    } finally {
      setMgrLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
          <p className="text-sm text-gray-500">Loading request...</p>
        </div>
      </div>
    );
  }

  if (error || !request) {
    return (
      <div className="text-center py-20">
        <p className="text-red-600">{error || 'Request not found'}</p>
        <Link to="/" className="mt-4 inline-block text-sm text-indigo-600 hover:underline">
          Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <Link to="/" className="text-sm text-indigo-600 hover:underline mb-2 inline-block">
            &larr; Back to Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">{request.title}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <StatusBadge status={request.status} />
            {request.priority_score !== null && (
              <span className="text-sm text-gray-600">
                Priority: <span className="font-semibold">{request.priority_score.toFixed(1)}</span>
              </span>
            )}
            <span className="text-sm text-gray-400">#{request.id}</span>
          </div>
        </div>
      </div>

      {/* Request Info */}
      <div className={cardClass}>
        <h2 className={sectionTitle}>Request Information</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
          <div>
            <p className={labelClass}>Requester</p>
            <p className={valueClass}>{request.requester_name}</p>
          </div>
          <div>
            <p className={labelClass}>Category</p>
            <p className={valueClass}>{formatLabel(request.category)}</p>
          </div>
          <div>
            <p className={labelClass}>Urgency</p>
            <p className={valueClass}>{request.urgency} / 5</p>
          </div>
          <div>
            <p className={labelClass}>Business Impact</p>
            <p className={valueClass}>{request.business_impact} / 5</p>
          </div>
          <div>
            <p className={labelClass}>Assigned Team</p>
            <p className={valueClass}>{request.assigned_team || '—'}</p>
          </div>
          <div>
            <p className={labelClass}>Assigned Owner</p>
            <p className={valueClass}>{request.assigned_owner || '—'}</p>
          </div>
          <div>
            <p className={labelClass}>Desired Completion</p>
            <p className={valueClass}>{formatShortDate(request.desired_completion_date)}</p>
          </div>
          <div>
            <p className={labelClass}>Created</p>
            <p className={valueClass}>{formatDate(request.created_at)}</p>
          </div>
        </div>
        <div className="mt-6">
          <p className={labelClass}>Description</p>
          <p className="mt-1 text-sm text-gray-700 whitespace-pre-wrap">{request.description}</p>
        </div>
      </div>

      {/* Routing Decision */}
      {request.routing_decision && (
        <div className={cardClass}>
          <h2 className={sectionTitle}>
            <span className="flex items-center gap-2">
              <svg className="h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
              </svg>
              AI Routing Decision
            </span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-4">
            <div>
              <p className={labelClass}>Suggested Team</p>
              <p className={valueClass}>{request.routing_decision.suggested_team}</p>
            </div>
            <div>
              <p className={labelClass}>Priority Score</p>
              <p className={valueClass}>{request.routing_decision.priority_score.toFixed(1)}</p>
            </div>
            <div>
              <p className={labelClass}>Category Match</p>
              <p className={valueClass}>{formatLabel(request.routing_decision.category_match)}</p>
            </div>
          </div>
          <div>
            <p className={labelClass}>Routing Explanation</p>
            <p className="mt-1 text-sm text-gray-700">{request.routing_decision.routing_explanation}</p>
          </div>
        </div>
      )}

      {/* AI Summary */}
      <div className={cardClass}>
        <h2 className={sectionTitle}>
          <span className="flex items-center gap-2">
            <svg className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
            </svg>
            AI Summary
          </span>
        </h2>
        {request.ai_summary ? (
          <div className="space-y-4">
            <div>
              <p className={labelClass}>Summary</p>
              <p className="mt-1 text-sm text-gray-700">{request.ai_summary.summary}</p>
            </div>
            <div>
              <p className={labelClass}>Business Impact</p>
              <p className="mt-1 text-sm text-gray-700">{request.ai_summary.business_impact_explanation}</p>
            </div>
            <div>
              <p className={labelClass}>Recommended Action</p>
              <p className="mt-1 text-sm text-gray-700">{request.ai_summary.recommended_action}</p>
            </div>
            <div>
              <p className={labelClass}>Leadership Summary</p>
              <p className="mt-1 text-sm text-gray-700">{request.ai_summary.leadership_summary}</p>
            </div>
            {request.ai_summary.implementation_notes && (
              <div>
                <p className={labelClass}>Implementation Notes</p>
                <p className="mt-1 text-sm text-gray-700">{request.ai_summary.implementation_notes}</p>
              </div>
            )}
            <p className="text-xs text-gray-400">
              Generated by {request.ai_summary.provider_used} on {formatDate(request.ai_summary.created_at)}
            </p>
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-sm text-gray-500 mb-4">No AI summary has been generated yet.</p>
            {summaryError && (
              <p className="text-sm text-red-600 mb-3">{summaryError}</p>
            )}
            <button
              onClick={handleSummarize}
              disabled={summarizing}
              className="inline-flex items-center gap-2 rounded-md bg-purple-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {summarizing ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Generating...
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                  </svg>
                  Generate AI Summary
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Status History */}
      {request.updates.length > 0 && (
        <div className={cardClass}>
          <h2 className={sectionTitle}>Status History</h2>
          <div className="space-y-0">
            {request.updates.map((update, idx) => (
              <div key={update.id} className="relative flex gap-4 pb-6">
                {idx < request.updates.length - 1 && (
                  <div className="absolute left-[11px] top-6 bottom-0 w-0.5 bg-gray-200" />
                )}
                <div className="flex-shrink-0 mt-1">
                  <div className="h-6 w-6 rounded-full bg-indigo-100 flex items-center justify-center">
                    <div className="h-2 w-2 rounded-full bg-indigo-600" />
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-gray-900">{update.author_name}</span>
                    {update.status_change && (
                      <StatusBadge status={update.status_change} />
                    )}
                    <span className="text-xs text-gray-400">{formatDate(update.created_at)}</span>
                  </div>
                  {update.note && (
                    <p className="mt-1 text-sm text-gray-600">{update.note}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Manager Actions */}
      {isManager && (
        <div className={`${cardClass} border-indigo-200`}>
          <h2 className={sectionTitle}>
            <span className="flex items-center gap-2">
              <svg className="h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17l-5.1-5.1m0 0L11.42 4.97m-5.1 5.1H21M3 3v18" />
              </svg>
              Manager Actions
            </span>
          </h2>

          {mgrError && (
            <div className="mb-4 rounded-md bg-red-50 border border-red-200 p-3">
              <p className="text-sm text-red-700">{mgrError}</p>
            </div>
          )}

          <form onSubmit={handleManagerUpdate} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select
                  value={mgrStatus}
                  onChange={(e) => setMgrStatus(e.target.value)}
                  className="block w-full rounded-md border border-gray-300 px-3 py-2.5 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {statuses.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Assigned Owner</label>
                <input
                  type="text"
                  value={mgrOwner}
                  onChange={(e) => setMgrOwner(e.target.value)}
                  className="block w-full rounded-md border border-gray-300 px-3 py-2.5 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  placeholder="Owner name"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Note</label>
              <textarea
                rows={3}
                value={mgrNote}
                onChange={(e) => setMgrNote(e.target.value)}
                className="block w-full rounded-md border border-gray-300 px-3 py-2.5 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Add a note about this update..."
              />
            </div>
            <button
              type="submit"
              disabled={mgrLoading}
              className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {mgrLoading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Updating...
                </>
              ) : (
                'Update Request'
              )}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
