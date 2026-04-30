import { type FormEvent, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { fetchUsers } from '../api/auth';
import { createTask } from '../api/tasks';
import { ApiError } from '../api/client';
import ErrorMessage from '../components/ErrorMessage';
import LoadingState from '../components/LoadingState';
import type { UserSummary } from '../types/auth';
import { TASK_PRIORITIES, type TaskPriority } from '../types/task';
import { errorMessage } from '../utils/format';

export default function CreateTaskPage() {
  const navigate = useNavigate();
  const [users, setUsers] = useState<UserSummary[] | null>(null);
  const [usersError, setUsersError] = useState<string | null>(null);

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState<TaskPriority>('medium');
  const [assignee, setAssignee] = useState<number | ''>('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchUsers();
        if (!cancelled) setUsers(data);
      } catch (err) {
        if (!cancelled) setUsersError(errorMessage(err, 'Failed to load users.'));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const titleInvalid = touched && !title.trim();
  const assigneeInvalid = touched && !assignee;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched(true);
    setError(null);
    if (!title.trim() || !assignee) {
      return;
    }
    setSubmitting(true);
    try {
      const created = await createTask({
        title: title.trim(),
        description,
        priority,
        assigned_to_user_id: Number(assignee),
      });
      navigate(`/tasks/${created.id}`);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to create the task. Please try again.');
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (usersError) {
    return <ErrorMessage message={usersError} />;
  }

  if (!users) {
    return <LoadingState message="Loading users..." />;
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Create task</h2>
          <p>Capture a new business workflow task and assign it to an owner.</p>
        </div>
        <Link to="/tasks">
          <button type="button">Cancel</button>
        </Link>
      </div>
      <section className="card">
        <form className="form-grid" onSubmit={handleSubmit} noValidate aria-label="Create task form">
          <ErrorMessage message={error} />
          <div>
            <label htmlFor="title">Title</label>
            <input
              id="title"
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              required
              aria-invalid={titleInvalid}
              aria-describedby={titleInvalid ? 'title-error' : undefined}
            />
            {titleInvalid && (
              <div id="title-error" className="field-error">
                Title is required.
              </div>
            )}
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
                aria-invalid={assigneeInvalid}
                aria-describedby={assigneeInvalid ? 'assignee-error' : undefined}
              >
                <option value="">Select a user</option>
                {users.map((user) => (
                  <option key={user.id} value={user.id}>
                    {user.name} ({user.role})
                  </option>
                ))}
              </select>
              {assigneeInvalid && (
                <div id="assignee-error" className="field-error">
                  Please choose an assignee.
                </div>
              )}
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="primary" disabled={submitting}>
              {submitting ? 'Creating...' : 'Create task'}
            </button>
            <Link to="/tasks">
              <button type="button">Cancel</button>
            </Link>
          </div>
        </form>
      </section>
    </>
  );
}
