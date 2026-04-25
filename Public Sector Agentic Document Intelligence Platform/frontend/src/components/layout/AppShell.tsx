import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function AppShell() {
  const loc = useLocation();
  return (
    <div className="grid h-screen grid-cols-[18rem_1fr]">
      <Sidebar />
      <main className="overflow-y-auto">
        <div
          key={loc.pathname}
          className="mx-auto max-w-column px-12 py-12 doc-spine pl-16"
        >
          <Outlet />
        </div>
        <footer className="mx-auto max-w-column px-12 py-8 mt-12 border-t border-hair border-rule-soft">
          <p className="datum text-2xs text-ink-40 uppercase tracking-rubric">
            Filed under PublicSector ADIP — internal use — audit trail enabled
          </p>
        </footer>
      </main>
    </div>
  );
}
