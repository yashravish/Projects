import type { ProcessRequest } from '../types';
import StatusBadge from './StatusBadge';
import { formatLabel } from './FilterBar';

interface RequestTableProps {
  requests: ProcessRequest[];
  onRowClick?: (id: number) => void;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function priorityColor(score: number | null): string {
  if (score === null) return 'text-gray-400';
  if (score >= 8) return 'text-red-600 font-semibold';
  if (score >= 5) return 'text-yellow-600 font-semibold';
  return 'text-green-600';
}

export default function RequestTable({ requests, onRowClick }: RequestTableProps) {
  if (requests.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
        <svg className="mx-auto h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12H9.75m10.5-9v6.375c0 .621-.504 1.125-1.125 1.125H4.875A1.125 1.125 0 013.75 15V5.625c0-.621.504-1.125 1.125-1.125h5.25" />
        </svg>
        <p className="mt-4 text-sm text-gray-500">No requests found</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Title</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Priority</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Team</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {requests.map((r, idx) => (
              <tr
                key={r.id}
                onClick={() => onRowClick?.(r.id)}
                className={`${onRowClick ? 'cursor-pointer' : ''} ${
                  idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'
                } hover:bg-indigo-50/50 transition-colors`}
              >
                <td className="px-4 py-3 text-sm text-gray-500 font-mono">#{r.id}</td>
                <td className="px-4 py-3 text-sm text-gray-900 font-medium max-w-xs truncate">{r.title}</td>
                <td className="px-4 py-3 text-sm text-gray-600">{formatLabel(r.category)}</td>
                <td className="px-4 py-3"><StatusBadge status={r.status} /></td>
                <td className={`px-4 py-3 text-sm ${priorityColor(r.priority_score)}`}>
                  {r.priority_score !== null ? r.priority_score.toFixed(1) : '—'}
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">{r.assigned_team || '—'}</td>
                <td className="px-4 py-3 text-sm text-gray-500">{formatDate(r.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
