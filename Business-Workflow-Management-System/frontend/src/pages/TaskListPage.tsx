import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { listTasks } from '../api/tasks';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import PriorityBadge from '../components/PriorityBadge';
import { errorMessage, formatDateTime } from '../utils/format';
import {
  TASK_PRIORITIES,
  TASK_STATUSES,
  type Task,
  type TaskPriority,
  type TaskStatus,
} from '../types/task';

export default function TaskListPage() {
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<TaskStatus | ''>('');
  const [priorityFilter, setPriorityFilter] = useState<TaskPriority | ''>('');
  const [search, setSearch] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await listTasks({
          status: statusFilter || undefined,
          priority: priorityFilter || undefined,
        });
        if (!cancelled) setTasks(data);
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, 'Failed to load tasks'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [statusFilter, priorityFilter]);

  const filtered = useMemo(() => {
    if (!tasks) return [];
    if (!search.trim()) return tasks;
    const term = search.toLowerCase();
    return tasks.filter(
      (task) =>
        task.title.toLowerCase().includes(term) ||
        task.description.toLowerCase().includes(term) ||
        task.assignee.name.toLowerCase().includes(term)
    );
  }, [tasks, search]);

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Tasks</h2>
          <p>Browse all workflow tasks across the organization.</p>
        </div>
        <Link to="/tasks/new">
          <button type="button" className="primary">
            + New task
          </button>
        </Link>
      </div>

      <section className="filters" aria-label="Filters">
        <div>
          <label htmlFor="filter-status">Status</label>
          <select
            id="filter-status"
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value as TaskStatus | '')}
          >
            <option value="">All statuses</option>
            {TASK_STATUSES.map((s) => (
              <option key={s} value={s}>
                {s.replace('_', ' ')}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="filter-priority">Priority</label>
          <select
            id="filter-priority"
            value={priorityFilter}
            onChange={(event) => setPriorityFilter(event.target.value as TaskPriority | '')}
          >
            <option value="">All priorities</option>
            {TASK_PRIORITIES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="filter-search">Search</label>
          <input
            id="filter-search"
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search title, description, or owner"
          />
        </div>
      </section>

      {error && <ErrorMessage message={error} />}
      {loading ? (
        <LoadingState message="Loading tasks..." />
      ) : (
        <div className="table-wrapper">
          <table className="data-table" aria-label="Tasks">
            <thead>
              <tr>
                <th>Title</th>
                <th>Status</th>
                <th>Priority</th>
                <th>Assignee</th>
                <th>Created by</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={6} className="empty-state">
                    No tasks match your filters.
                  </td>
                </tr>
              ) : (
                filtered.map((task) => (
                  <tr key={task.id} data-testid="task-row">
                    <td>
                      <Link to={`/tasks/${task.id}`}>{task.title}</Link>
                    </td>
                    <td>
                      <StatusBadge status={task.status} />
                    </td>
                    <td>
                      <PriorityBadge priority={task.priority} />
                    </td>
                    <td>{task.assignee.name}</td>
                    <td>{task.creator.name}</td>
                    <td>{formatDateTime(task.created_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
