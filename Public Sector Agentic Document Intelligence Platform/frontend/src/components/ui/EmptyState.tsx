import { ReactNode } from 'react';

interface Props {
  rubric?: string;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}

export function EmptyState({ rubric, title, description, action }: Props) {
  return (
    <div className="border-hair border-rule-soft py-16 px-8 text-center">
      {rubric ? <p className="rubric mb-3">{rubric}</p> : null}
      <h2 className="display text-3xl mb-3">{title}</h2>
      {description ? (
        <p className="max-w-prose mx-auto text-ink-80 leading-relaxed">{description}</p>
      ) : null}
      {action ? <div className="mt-6 flex justify-center">{action}</div> : null}
    </div>
  );
}
