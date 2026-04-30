import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  approveTask,
  completeTask,
  deleteTask,
  getTask,
  getWorkflowEvents,
  rejectTask,
  submitTask,
} from '../api/tasks';
import { ApiError } from '../api/client';
import { useAuth } from '../auth/AuthContext';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import PriorityBadge from '../components/PriorityBadge';
import { errorMessage, formatDateTime, formatStatus } from '../utils/format';
import type { Task, WorkflowEvent } from '../types/task';

export default function TaskDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();

  const [task, setTask] = useState<Task | null>(null);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [comment, setComment] = useState('');

  const taskId = Number(id);

  const reload = useCallback(async () => {
    if (!Number.isFinite(taskId)) return;
    setLoading(true);
    setError(null);
    try {
      const [taskData, history] = await Promise.all([
        getTask(taskId),
        getWorkflowEvents(taskId),
      ]);
      setTask(taskData);
      setEvents(history);
    } catch (err) {
      setError(errorMessage(err, 'Failed to load task'));
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    reload();
  }, [reload]);

  async function runAction(action: () => Promise<Task>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setComment('');
      await reload();
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError('Action failed.');
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    if (!task) return;
    if (!window.confirm(`Delete task "${task.title}"?`)) return;
    setBusy(true);
    setError(null);
    try {
      await deleteTask(task.id);
      navigate('/tasks');
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError('Failed to delete.');
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <LoadingState message="Loading task..." />;
  if (error && !task) return <ErrorMessage message={error} />;
  if (!task || !user) return null;

  const isManager = user.role === 'manager';
  const isCreator = task.created_by_user_id === user.id;
  const canEdit = task.status !== 'completed' && (isManager || isCreator);
  const canDelete =
    task.status !== 'approved' && task.status !== 'completed' && (isManager || isCreator);
  const canSubmit =
    (task.status === 'pending' ||
      task.status === 'in_progress' ||
      task.status === 'rejected') &&
    (isCreator || isManager);
  const canApproveReject = isManager && task.status === 'submitted';
  const canComplete = isManager && task.status === 'approved';

  return (
    <>
      <div className="page-header">
        <div>
          <h2>{task.title}</h2>
          <p>Task #{task.id}</p>
        </div>
        <div className="button-row">
          <StatusBadge status={task.status} />
          <PriorityBadge priority={task.priority} />
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      <section className="card" aria-label="Task details">
        <dl className="kv">
          <dt>Description</dt>
          <dd>{task.description || <span className="meta">No description.</span>}</dd>
          <dt>Status</dt>
          <dd>
            <StatusBadge status={task.status} />
          </dd>
          <dt>Priority</dt>
          <dd>
            <PriorityBadge priority={task.priority} />
          </dd>
          <dt>Assigned to</dt>
          <dd>
            {task.assignee.name} ({task.assignee.email})
          </dd>
          <dt>Created by</dt>
          <dd>
            {task.creator.name} ({task.creator.email})
          </dd>
          <dt>Created</dt>
          <dd>{formatDateTime(task.created_at)}</dd>
          <dt>Updated</dt>
          <dd>{formatDateTime(task.updated_at)}</dd>
        </dl>
      </section>

      <section className="card" aria-label="Task actions">
        <h3>Actions</h3>
        <div className="form-grid">
          <div>
            <label htmlFor="action-comment">Comment (optional)</label>
            <input
              id="action-comment"
              type="text"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="Add an optional note for this action"
            />
          </div>
          <div className="button-row">
            {canEdit && (
              <Link to={`/tasks/${task.id}/edit`}>
                <button type="button">Edit</button>
              </Link>
            )}
            {canSubmit && (
              <button
                type="button"
                className="primary"
                disabled={busy}
                data-testid="submit-task"
                onClick={() => runAction(() => submitTask(task.id, comment || undefined))}
              >
                Submit for review
              </button>
            )}
            {canApproveReject && (
              <>
                <button
                  type="button"
                  className="success"
                  data-testid="approve-task"
                  disabled={busy}
                  onClick={() => runAction(() => approveTask(task.id, comment || undefined))}
                >
                  Approve
                </button>
                <button
                  type="button"
                  className="danger"
                  data-testid="reject-task"
                  disabled={busy}
                  onClick={() => runAction(() => rejectTask(task.id, comment || undefined))}
                >
                  Reject
                </button>
              </>
            )}
            {canComplete && (
              <button
                type="button"
                className="success"
                data-testid="complete-task"
                disabled={busy}
                onClick={() => runAction(() => completeTask(task.id, comment || undefined))}
              >
                Mark complete
              </button>
            )}
            {canDelete && (
              <button
                type="button"
                className="danger"
                disabled={busy}
                onClick={handleDelete}
              >
                Delete
              </button>
            )}
          </div>
        </div>
      </section>

      <section className="card" aria-label="Workflow history">
        <h3>Workflow history</h3>
        {events.length === 0 ? (
          <div className="empty-state">No workflow events yet.</div>
        ) : (
          <ol className="timeline">
            {events.map((event) => (
              <li key={event.id} className="timeline-item">
                <strong>
                  {event.from_status
                    ? `${formatStatus(event.from_status)} → ${formatStatus(event.to_status)}`
                    : `Created (${formatStatus(event.to_status)})`}
                </strong>
                <div className="meta">
                  {event.changed_by.name} ({event.changed_by.role}) ·{' '}
                  {formatDateTime(event.created_at)}
                </div>
                {event.note && <div>{event.note}</div>}
              </li>
            ))}
          </ol>
        )}
      </section>
    </>
  );
}
