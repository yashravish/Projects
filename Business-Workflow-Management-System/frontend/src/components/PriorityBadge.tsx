import type { TaskPriority } from '../types/task';

const LABELS: Record<TaskPriority, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
};

export default function PriorityBadge({ priority }: { priority: TaskPriority }) {
  return (
    <span className={`badge priority-${priority}`} aria-label={`Priority ${LABELS[priority]}`}>
      {LABELS[priority]}
    </span>
  );
}
