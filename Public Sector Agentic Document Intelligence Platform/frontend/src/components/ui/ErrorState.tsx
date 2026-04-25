import { ReactNode } from 'react';
import { AlertOctagon } from 'lucide-react';

interface Props {
  title?: string;
  description?: ReactNode;
  action?: ReactNode;
}

export function ErrorState({ title = 'Failure recorded', description, action }: Props) {
  return (
    <div className="border-hair border-seal/60 bg-seal/5 p-6">
      <div className="flex items-start gap-3">
        <AlertOctagon size={18} className="text-seal mt-0.5 shrink-0" aria-hidden />
        <div className="flex-1 min-w-0">
          <p className="rubric text-seal">incident</p>
          <h3 className="display text-xl mt-0.5 text-seal">{title}</h3>
          {description ? (
            <p className="mt-2 text-sm text-ink-80 leading-relaxed">{description}</p>
          ) : null}
          {action ? <div className="mt-4">{action}</div> : null}
        </div>
      </div>
    </div>
  );
}
