import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export default function NavBar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate('/login', { replace: true });
  }

  return (
    <header className="navbar" role="banner">
      <div className="navbar-brand-wrap">
        <h1 className="navbar-brand">Workflow Desk</h1>
        <span className="navbar-tagline">Operations</span>
      </div>
      <nav aria-label="Primary">
        <NavLink to="/dashboard" end>
          Dashboard
        </NavLink>
        <NavLink to="/tasks">Tasks</NavLink>
        <NavLink to="/tasks/new">Create Task</NavLink>
        <NavLink to="/reports">Reports</NavLink>
        <NavLink to="/audit-logs">Audit Logs</NavLink>
      </nav>
      <div className="navbar-user">
        {user && (
          <>
            <span data-testid="current-user">
              {user.name} <strong>({user.role})</strong>
            </span>
            <button type="button" className="ghost" onClick={handleLogout}>
              Sign out
            </button>
          </>
        )}
      </div>
    </header>
  );
}
