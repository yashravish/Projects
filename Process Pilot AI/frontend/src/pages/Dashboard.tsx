import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AnalyticsOverview, ProcessRequest } from '../types';
import { getAnalyticsOverview, getRequests } from '../api/client';
import StatCard from '../components/StatCard';
import FilterBar from '../components/FilterBar';
import RequestTable from '../components/RequestTable';

export default function Dashboard() {
  const navigate = useNavigate();
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [requests, setRequests] = useState<ProcessRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ department: '', category: '', status: '' });

  useEffect(() => {
    setLoading(true);
    setError('');
    Promise.all([getAnalyticsOverview(), getRequests(filters)])
      .then(([ov, reqs]) => {
        setOverview(ov);
        setRequests(reqs);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load data'))
      .finally(() => setLoading(false));
  }, [filters]);

  if (loading && !overview) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
          <p className="text-sm text-gray-500">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">Overview of all process modernization requests</p>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {overview && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <StatCard title="Total Requests" value={overview.total_requests} color="indigo" />
          <StatCard title="Open" value={overview.open_requests} color="blue" />
          <StatCard title="Closed" value={overview.closed_requests} color="green" />
          <StatCard title="Avg Priority" value={overview.avg_priority.toFixed(1)} color="yellow" />
          <StatCard title="This Week" value={overview.requests_this_week} color="purple" />
        </div>
      )}

      <FilterBar filters={filters} onChange={setFilters} />

      <RequestTable
        requests={requests}
        onRowClick={(id) => navigate(`/requests/${id}`)}
      />
    </div>
  );
}
