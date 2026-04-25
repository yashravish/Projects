import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { RequireAuth } from '@/components/RequireAuth';
import { AuditPage } from '@/pages/AuditPage';
import { DocumentsPage } from '@/pages/DocumentsPage';
import { EvaluationsPage } from '@/pages/EvaluationsPage';
import { InquiryPage } from '@/pages/InquiryPage';
import { LoginPage } from '@/pages/LoginPage';
import { ModelsPage } from '@/pages/ModelsPage';
import { RegisterPage } from '@/pages/RegisterPage';

export const router = createBrowserRouter([
  { path: '/login', element: <LoginPage /> },
  { path: '/register', element: <RegisterPage /> },
  {
    path: '/',
    element: (
      <RequireAuth>
        <AppShell />
      </RequireAuth>
    ),
    children: [
      { index: true, element: <Navigate to="/documents" replace /> },
      { path: 'documents', element: <DocumentsPage /> },
      { path: 'query', element: <InquiryPage /> },
      { path: 'evaluations', element: <EvaluationsPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'audit', element: <AuditPage /> },
    ],
  },
  { path: '*', element: <Navigate to="/" replace /> },
]);
