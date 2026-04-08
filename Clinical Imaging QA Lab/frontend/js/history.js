/**
 * History page logic — loads and renders capture history table
 * with status badges and retry buttons for failed captures.
 */
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
});

async function loadHistory() {
  const tbody = document.getElementById('history-body');
  const alertEl = document.getElementById('history-alert');

  tbody.innerHTML = '<tr><td colspan="9" class="loading-overlay"><span class="spinner"></span> Loading history…</td></tr>';

  try {
    const captures = await API.getCaptures();
    alertEl.hidden = true;
    renderHistory(tbody, captures);
  } catch (err) {
    tbody.innerHTML = '';
    alertEl.textContent = 'Failed to load capture history: ' + err.message;
    alertEl.className = 'alert alert-danger';
    alertEl.hidden = false;
    alertEl.setAttribute('role', 'alert');
  }
}

function renderHistory(tbody, captures) {
  if (!captures || captures.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="table-empty">No capture records found</td></tr>';
    return;
  }

  tbody.innerHTML = captures.map(c => `
    <tr>
      <td>${c.id}</td>
      <td>${escapeHtml(c.patient_id)}</td>
      <td>${escapeHtml(c.session_id)}</td>
      <td>${c.image_type}</td>
      <td>
        <span class="badge badge-${statusBadge(c.capture_status)}">${c.capture_status}</span>
      </td>
      <td>${c.device_name || '—'}</td>
      <td>${c.retry_count}</td>
      <td>${formatDate(c.created_at)}</td>
      <td>
        ${c.capture_status === 'failed'
          ? `<button class="btn btn-sm btn-secondary" onclick="retryFromHistory(${c.id})" aria-label="Retry capture ${c.id}">Retry</button>`
          : '—'}
      </td>
    </tr>
  `).join('');
}

async function retryFromHistory(captureId) {
  const alertEl = document.getElementById('history-alert');
  try {
    alertEl.innerHTML = '<span class="spinner"></span> Retrying capture…';
    alertEl.className = 'alert alert-info';
    alertEl.hidden = false;

    await API.retryCapture(captureId);
    alertEl.hidden = true;
    loadHistory();
  } catch (err) {
    alertEl.textContent = 'Retry failed: ' + err.message;
    alertEl.className = 'alert alert-danger';
    alertEl.hidden = false;
  }
}

function statusBadge(status) {
  const map = { success: 'success', failed: 'danger', pending: 'warning' };
  return map[status] || 'neutral';
}

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}
