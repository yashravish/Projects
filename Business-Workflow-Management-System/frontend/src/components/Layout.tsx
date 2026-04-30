import type { ReactNode } from 'react';
import NavBar from './NavBar';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <NavBar />
      <main className="page" role="main">
        {children}
      </main>
    </div>
  );
}
