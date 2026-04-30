import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { listAuditLogs } from '../api/auditLogs';
import { listTasks } from '../api/tasks';
import {
  getTasksByPriority,
  getTasksByStatus,
} from '../api/reports';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import PriorityBadge from '../components/PriorityBadge';
import { errorMessage, formatDateTime, formatStatus } from '../utils/format';
import type { AuditLog } from '../types/auditLog';
import type { Task } from '../types/task';
import type { TasksByPriorityReport, TasksByStatusReport } from '../types/report';

interface DashboardData {
  tasks: Task[];
  myTasks: Task[];
  byStatus: TasksByStatusReport;
  byPriority: TasksByPriorityReport;
  recentLogs: AuditLog[];
}

export default function DashboardPage() {
  const { user } = useAuth();
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!user) return;
      setLoading(true);
      setError(null);
      try {
        const [tasks, myTasks, byStatus, byPriority, recentLogs] = await Promise.all([
          listTasks(),
          listTasks({ assigned_to_user_id: user.id }),
          getTasksByStatus(),
          getTasksByPriority(),
          listAuditLogs({ limit: 8 }),
        ]);
        if (cancelled) return;
        setData({ tasks, myTasks, byStatus, byPriority, recentLogs });
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, 'Failed to load dashboard.'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [user]);

  if (loading) return <LoadingState message="Loading dashboard..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!data || !user) return null;

  const totalTasks = data.byStatus.total;
  const submittedCount =
    data.byStatus.items.find((item) => item.status === 'submitted')?.count ?? 0;
  const approvedCount =
    data.byStatus.items.find((item) => item.status === 'approved')?.count ?? 0;
  const completedCount =
    data.byStatus.items.find((item) => item.status === 'completed')?.count ?? 0;

  const recentTasks = data.tasks.slice(0, 5);

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Dashboard</h2>
          <p>
            Welcome back, {user.name}. Here is the current state of business workflows.
          </p>
        </div>
        <Link to="/tasks/new" className="primary" role="button" style={{ alignSelf: 'center' }}>
          <button className="primary" type="button">
            + New task
          </button>
        </Link>
      </div>

      <section className="summary-grid" data-testid="summary-cards" aria-label="Workflow summary">
        <div className="summary-card">
          <div className="label">Total tasks</div>
          <div className="value" data-testid="metric-total">{totalTasks}</div>
        </div>
        <div className="summary-card">
          <div className="label">Awaiting approval</div>
          <div className="value">{submittedCount}</div>
        </div>
        <div className="summary-card">
          <div className="label">Approved</div>
          <div className="value">{approvedCount}</div>
        </div>
        <div className="summary-card">
          <div className="label">Completed</div>
          <div className="value">{completedCount}</div>
        </div>
        <div className="summary-card">
          <div className="label">Assigned to me</div>
          <div className="value">{data.myTasks.length}</div>
        </div>
      </section>

      <div className="split">
        <section className="card" aria-label="Tasks by status">
          <h3>Tasks by status</h3>
          <div className="metric-row">
            {data.byStatus.items.map((item) => (
              <span key={item.status} className="metric-pill">
                {formatStatus(item.status)}: {item.count}
              </span>
            ))}
          </div>
        </section>
        <section className="card" aria-label="Tasks by priority">
          <h3>Tasks by priority</h3>
          <div className="metric-row">
            {data.byPriority.items.map((item) => (
              <span key={item.priority} className="metric-pill">
                {formatStatus(item.priority)}: {item.count}
              </span>
            ))}
          </div>
        </section>
      </div>

      <div className="split">
        <section className="card" aria-label="My assigned tasks">
          <h3>My assigned tasks</h3>
          {data.myTasks.length === 0 ? (
            <div className="empty-state">Nothing assigned to you right now.</div>
          ) : (
            <div className="list-rows">
              {data.myTasks.slice(0, 5).map((task) => (
                <div key={task.id} className="list-row">
                  <div>
                    <Link to={`/tasks/${task.id}`}>{task.title}</Link>
                    <div className="meta">{formatDateTime(task.updated_at)}</div>
                  </div>
                  <div className="button-row">
                    <PriorityBadge priority={task.priority} />
                    <StatusBadge status={task.status} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card" aria-label="Recent tasks">
          <h3>Recent tasks</h3>
          {recentTasks.length === 0 ? (
            <div className="empty-state">No tasks yet.</div>
          ) : (
            <div className="list-rows">
              {recentTasks.map((task) => (
                <div key={task.id} className="list-row">
                  <div>
                    <Link to={`/tasks/${task.id}`}>{task.title}</Link>
                    <div className="meta">
                      Assigned to {task.assignee.name} ·{' '}
                      {formatDateTime(task.created_at)}
                    </div>
                  </div>
                  <StatusBadge status={task.status} />
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <section className="card" aria-label="Recent audit activity">
        <h3>Recent audit activity</h3>
        {data.recentLogs.length === 0 ? (
          <div className="empty-state">No audit activity yet.</div>
        ) : (
          <div className="list-rows">
            {data.recentLogs.map((log) => (
              <div key={log.id} className="list-row">
                <div>
                  <strong>{formatStatus(log.action)}</strong>
                  <div className="meta">
                    {log.user ? `${log.user.name} (${log.user.role})` : 'System'} ·{' '}
                    {log.entity_type} #{log.entity_id ?? '-'}
                  </div>
                </div>
                <div className="meta">{formatDateTime(log.created_at)}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
