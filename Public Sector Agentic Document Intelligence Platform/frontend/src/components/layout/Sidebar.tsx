import { NavLink } from 'react-router-dom';
import { FileText, Search, Activity, Anvil, Shield } from 'lucide-react';
import { useAuth } from '@/state/auth';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/cn';

interface NavItem {
  to: string;
  label: string;
  rubric: string;
  icon: typeof FileText;
}

const NAV: NavItem[] = [
  { to: '/documents', label: 'Documents', rubric: '002', icon: FileText },
  { to: '/query', label: 'Inquiry', rubric: '003', icon: Search },
  { to: '/evaluations', label: 'Evaluations', rubric: '004', icon: Activity },
  { to: '/models', label: 'The Forge', rubric: '005', icon: Anvil },
  { to: '/audit', label: 'Audit', rubric: '006', icon: Shield },
];

export function Sidebar() {
  const { user, logout } = useAuth();
  return (
    <aside className="flex h-full flex-col border-r border-hair border-rule-soft bg-paper-deep">
      <div className="px-7 pt-9 pb-6">
        <p className="rubric">001</p>
        <h1 className="display text-2xl mt-1 leading-none">
          Public<span className="display-italic">Sector</span>
          <br />
          ADIP
        </h1>
        <p className="datum text-2xs text-ink-60 mt-3">
          AGENTIC&nbsp;·&nbsp;DOCUMENT&nbsp;·&nbsp;INTELLIGENCE
        </p>
      </div>

      <hr className="rule" />

      <nav className="flex-1 py-4">
        <ul>
          {NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  cn(
                    'group flex items-baseline gap-4 px-7 py-3 transition-colors duration-200',
                    isActive
                      ? 'bg-paper text-ink'
                      : 'text-ink-60 hover:text-ink hover:bg-paper/60',
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={cn(
                        'datum text-2xs uppercase tracking-rubric w-12',
                        isActive ? 'text-seal' : 'text-ink-40',
                      )}
                    >
                      {item.rubric}
                    </span>
                    <span
                      className={cn(
                        'flex-1 font-medium tracking-[-0.005em]',
                        isActive && 'border-b-hair border-rule pb-0.5',
                      )}
                    >
                      {item.label}
                    </span>
                    <item.icon
                      size={15}
                      strokeWidth={1.5}
                      className={cn(
                        'transition-opacity',
                        isActive ? 'opacity-100 text-seal' : 'opacity-50',
                      )}
                      aria-hidden
                    />
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <hr className="rule-soft" />

      <div className="p-7">
        {user ? (
          <>
            <p className="rubric">analyst</p>
            <p className="text-sm mt-0.5 break-all">{user.email}</p>
            <p className="datum text-2xs text-ink-60 mt-1 uppercase tracking-rubric">
              {user.role}
            </p>
            <div className="mt-5">
              <Button variant="ghost" onClick={logout}>
                Sign out
              </Button>
            </div>
          </>
        ) : null}
      </div>
    </aside>
  );
}
