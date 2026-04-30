import { useEffect, useState } from 'react';
import {
  exportTasksCsv,
  getApprovalSummary,
  getTasksByPriority,
  getTasksByStatus,
  getUserWorkload,
} from '../api/reports';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { errorMessage, formatStatus, titleCase } from '../utils/format';
import type {
  ApprovalSummaryReport,
  TasksByPriorityReport,
  TasksByStatusReport,
  UserWorkloadReport,
} from '../types/report';

interface ReportBundle {
  byStatus: TasksByStatusReport;
  byPriority: TasksByPriorityReport;
  workload: UserWorkloadReport;
  approval: ApprovalSummaryReport;
}

export default function ReportsPage() {
  const [data, setData] = useState<ReportBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [byStatus, byPriority, workload, approval] = await Promise.all([
          getTasksByStatus(),
          getTasksByPriority(),
          getUserWorkload(),
          getApprovalSummary(),
        ]);
        if (cancelled) return;
        setData({ byStatus, byPriority, workload, approval });
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, 'Failed to load reports'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleExport() {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await exportTasksCsv();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'tasks.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(errorMessage(err, 'Failed to export CSV'));
    } finally {
      setExporting(false);
    }
  }

  if (loading) return <LoadingState message="Loading reports..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Reports</h2>
          <p>Operational metrics and exports for managers and analysts.</p>
        </div>
        <div className="button-row">
          <button
            type="button"
            className="primary"
            onClick={handleExport}
            disabled={exporting}
            data-testid="export-csv"
          >
            {exporting ? 'Exporting...' : 'Export tasks CSV'}
          </button>
        </div>
      </div>

      {exportError && <ErrorMessage message={exportError} />}

      <section className="card" aria-label="Tasks by status">
        <h3>Tasks by status</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {data.byStatus.items.map((item) => (
                <tr key={item.status}>
                  <td>{formatStatus(item.status)}</td>
                  <td>{item.count}</td>
                </tr>
              ))}
              <tr>
                <td>
                  <strong>Total</strong>
                </td>
                <td>
                  <strong>{data.byStatus.total}</strong>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" aria-label="Tasks by priority">
        <h3>Tasks by priority</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Priority</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {data.byPriority.items.map((item) => (
                <tr key={item.priority}>
                  <td>{titleCase(item.priority)}</td>
                  <td>{item.count}</td>
                </tr>
              ))}
              <tr>
                <td>
                  <strong>Total</strong>
                </td>
                <td>
                  <strong>{data.byPriority.total}</strong>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" aria-label="User workload">
        <h3>User workload</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>Total</th>
                <th>Pending</th>
                <th>In progress</th>
                <th>Submitted</th>
                <th>Approved</th>
                <th>Rejected</th>
                <th>Completed</th>
              </tr>
            </thead>
            <tbody>
              {data.workload.items.map((item) => (
                <tr key={item.user_id}>
                  <td>{item.name}</td>
                  <td>{titleCase(item.role)}</td>
                  <td>{item.total}</td>
                  <td>{item.pending}</td>
                  <td>{item.in_progress}</td>
                  <td>{item.submitted}</td>
                  <td>{item.approved}</td>
                  <td>{item.rejected}</td>
                  <td>{item.completed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" aria-label="Approval summary">
        <h3>Approval summary</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Decision</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {data.approval.items.map((item) => (
                <tr key={item.decision}>
                  <td>{titleCase(item.decision)}</td>
                  <td>{item.count}</td>
                </tr>
              ))}
              <tr>
                <td>
                  <strong>Total</strong>
                </td>
                <td>
                  <strong>{data.approval.total}</strong>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
