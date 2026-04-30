import type { TaskStatus } from '../types/task';

const LABELS: Record<TaskStatus, string> = {
  pending: 'Pending',
  in_progress: 'In progress',
  submitted: 'Submitted',
  approved: 'Approved',
  rejected: 'Rejected',
  completed: 'Completed',
};

export default function StatusBadge({ status }: { status: TaskStatus }) {
  return (
    <span className={`badge status-${status}`} aria-label={`Status ${LABELS[status]}`}>
      {LABELS[status]}
    </span>
  );
}
