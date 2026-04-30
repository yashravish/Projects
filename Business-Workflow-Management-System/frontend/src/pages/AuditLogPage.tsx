import { useEffect, useState } from 'react';
import { listAuditLogs } from '../api/auditLogs';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { errorMessage, formatDateTime, formatStatus } from '../utils/format';
import { AUDIT_ACTIONS, type AuditAction, type AuditLog } from '../types/auditLog';

export default function AuditLogPage() {
  const [logs, setLogs] = useState<AuditLog[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionFilter, setActionFilter] = useState<AuditAction | ''>('');

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await listAuditLogs({
          action: actionFilter || undefined,
          limit: 200,
        });
        if (!cancelled) setLogs(data);
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, 'Failed to load logs'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [actionFilter]);

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Audit logs</h2>
          <p>Every meaningful action recorded across the workflow system.</p>
        </div>
      </div>

      <section className="filters" aria-label="Audit filters">
        <div>
          <label htmlFor="action-filter">Action</label>
          <select
            id="action-filter"
            value={actionFilter}
            onChange={(event) => setActionFilter(event.target.value as AuditAction | '')}
          >
            <option value="">All actions</option>
            {AUDIT_ACTIONS.map((action) => (
              <option key={action} value={action}>
                {formatStatus(action)}
              </option>
            ))}
          </select>
        </div>
      </section>

      {error && <ErrorMessage message={error} />}
      {loading ? (
        <LoadingState message="Loading audit logs..." />
      ) : (
        <div className="table-wrapper">
          <table className="data-table" aria-label="Audit log entries">
            <thead>
              <tr>
                <th>When</th>
                <th>User</th>
                <th>Action</th>
                <th>Entity</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {logs && logs.length > 0 ? (
                logs.map((log) => (
                  <tr key={log.id}>
                    <td>{formatDateTime(log.created_at)}</td>
                    <td>
                      {log.user
                        ? `${log.user.name} (${log.user.role})`
                        : <span className="meta">System</span>}
                    </td>
                    <td>{formatStatus(log.action)}</td>
                    <td>
                      {log.entity_type}
                      {log.entity_id ? ` #${log.entity_id}` : ''}
                    </td>
                    <td>{log.details ?? <span className="meta">-</span>}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="empty-state">
                    No audit log entries.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
