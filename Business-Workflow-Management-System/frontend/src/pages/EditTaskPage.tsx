import { type FormEvent, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { fetchUsers } from '../api/auth';
import { getTask, updateTask } from '../api/tasks';
import { ApiError } from '../api/client';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import type { UserSummary } from '../types/auth';
import { TASK_PRIORITIES, type Task, type TaskPriority } from '../types/task';
import { errorMessage } from '../utils/format';

export default function EditTaskPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [task, setTask] = useState<Task | null>(null);
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState<TaskPriority>('medium');
  const [assignee, setAssignee] = useState<number | ''>('');
  const [submitting, setSubmitting] = useState(false);

  const taskId = Number(id);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!Number.isFinite(taskId)) return;
      setLoading(true);
      setError(null);
      try {
        const [taskData, userData] = await Promise.all([getTask(taskId), fetchUsers()]);
        if (cancelled) return;
        setTask(taskData);
        setUsers(userData);
        setTitle(taskData.title);
        setDescription(taskData.description);
        setPriority(taskData.priority);
        setAssignee(taskData.assigned_to_user_id);
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, 'Failed to load task'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!title.trim() || !assignee) {
      setError('Title and assignee are required.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await updateTask(taskId, {
        title: title.trim(),
        description,
        priority,
        assigned_to_user_id: Number(assignee),
      });
      navigate(`/tasks/${taskId}`);
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError('Failed to update task.');
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <LoadingState message="Loading task..." />;
  if (error && !task) return <ErrorMessage message={error} />;
  if (!task) return null;

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Edit task</h2>
          <p>Task #{task.id}</p>
        </div>
        <Link to={`/tasks/${task.id}`}>
          <button type="button">Cancel</button>
        </Link>
      </div>
      <section className="card">
        <form className="form-grid" onSubmit={handleSubmit} aria-label="Edit task form">
          <ErrorMessage message={error} />
          <div>
            <label htmlFor="title">Title</label>
            <input
              id="title"
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="description">Description</label>
            <textarea
              id="description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={5}
            />
          </div>
          <div className="form-row">
            <div>
              <label htmlFor="priority">Priority</label>
              <select
                id="priority"
                value={priority}
                onChange={(event) => setPriority(event.target.value as TaskPriority)}
              >
                {TASK_PRIORITIES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="assignee">Assigned to</label>
              <select
                id="assignee"
                value={assignee}
                onChange={(event) =>
                  setAssignee(event.target.value ? Number(event.target.value) : '')
                }
                required
              >
                <option value="">Select a user</option>
                {users.map((user) => (
                  <option key={user.id} value={user.id}>
                    {user.name} ({user.role})
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="primary" disabled={submitting}>
              {submitting ? 'Saving...' : 'Save changes'}
            </button>
            <Link to={`/tasks/${task.id}`}>
              <button type="button">Cancel</button>
            </Link>
          </div>
        </form>
      </section>
    </>
  );
}
