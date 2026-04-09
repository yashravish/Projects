import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ProcessRequest } from '../types';
import { getRequests } from '../api/client';
import { useAuth } from '../context/AuthContext';
import StatCard from '../components/StatCard';
import RequestTable from '../components/RequestTable';

export default function ManagerQueue() {
  const { isManager } = useAuth();
  const navigate = useNavigate();
  const [requests, setRequests] = useState<ProcessRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isManager) return;
    setLoading(true);
    Promise.all([
      getRequests({ status: 'in_review' }),
      getRequests({ status: 'in_progress' }),
    ])
      .then(([inReview, inProgress]) => {
        setRequests([...inReview, ...inProgress]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load queue'))
      .finally(() => setLoading(false));
  }, [isManager]);

  if (!isManager) {
    return (
      <div className="text-center py-20">
        <div className="mx-auto h-16 w-16 rounded-full bg-red-100 flex items-center justify-center mb-4">
          <svg className="h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Access Denied</h2>
        <p className="mt-2 text-sm text-gray-500">This page is only accessible to managers.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
          <p className="text-sm text-gray-500">Loading manager queue...</p>
        </div>
      </div>
    );
  }

  const inReviewCount = requests.filter((r) => r.status === 'in_review').length;
  const inProgressCount = requests.filter((r) => r.status === 'in_progress').length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Manager Queue</h1>
        <p className="mt-1 text-sm text-gray-500">Requests awaiting review or currently in progress</p>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard title="In Review" value={inReviewCount} color="yellow" />
        <StatCard title="In Progress" value={inProgressCount} color="purple" />
        <StatCard title="Total in Queue" value={requests.length} color="indigo" />
      </div>

      <RequestTable
        requests={requests}
        onRowClick={(id) => navigate(`/requests/${id}`)}
      />
    </div>
  );
}
