/**
 * Dashboard page logic — loads summary stats, device status,
 * recent captures and recent defects on page load.
 */
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
});

async function loadDashboard() {
  const statusEl  = document.getElementById('device-status-area');
  const statsArea = document.getElementById('stats-area');
  const capturesBody = document.getElementById('recent-captures-body');
  const defectsBody  = document.getElementById('recent-defects-body');
  const alertEl = document.getElementById('dashboard-alert');

  try {
    const summary = await API.getDashboardSummary();

    renderDeviceStatus(statusEl, summary.device_status);
    renderStats(statsArea, summary);
    renderRecentCaptures(capturesBody, summary.recent_captures);
    renderRecentDefects(defectsBody, summary.recent_defects);
    alertEl.hidden = true;
  } catch (err) {
    alertEl.textContent = 'Failed to load dashboard data: ' + err.message;
    alertEl.className = 'alert alert-danger';
    alertEl.hidden = false;
    alertEl.setAttribute('role', 'alert');
  }
}

function renderDeviceStatus(container, status) {
  const dotClass = status === 'online' ? 'online' : (status === 'offline' ? 'offline' : 'unknown');
  const label = status.charAt(0).toUpperCase() + status.slice(1);
  container.innerHTML = `
    <div class="device-status" role="status" aria-live="polite">
      <span class="device-dot ${dotClass}" aria-hidden="true"></span>
      <div class="device-info">
        <h3>SIMULATED_SCANNER_01</h3>
        <p>Status: <span class="badge badge-${dotClass}">${label}</span></p>
      </div>
    </div>
  `;
}

function renderStats(container, s) {
  container.innerHTML = `
    <div class="stat-card">
      <div class="stat-icon primary" aria-hidden="true">📷</div>
      <div class="stat-content">
        <h3>${s.total_captures}</h3>
        <p>Total Captures</p>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon success" aria-hidden="true">✓</div>
      <div class="stat-content">
        <h3>${s.successful_captures}</h3>
        <p>Successful</p>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon danger" aria-hidden="true">✕</div>
      <div class="stat-content">
        <h3>${s.failed_captures}</h3>
        <p>Failed</p>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon warning" aria-hidden="true">🐛</div>
      <div class="stat-content">
        <h3>${s.open_defects}</h3>
        <p>Open Defects</p>
      </div>
    </div>
  `;
}

function renderRecentCaptures(tbody, captures) {
  if (!captures || captures.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No captures yet</td></tr>';
    return;
  }
  tbody.innerHTML = captures.map(c => `
    <tr>
      <td>${c.patient_id}</td>
      <td>${c.image_type}</td>
      <td><span class="badge badge-${c.capture_status === 'success' ? 'success' : (c.capture_status === 'failed' ? 'danger' : 'warning')}">${c.capture_status}</span></td>
      <td>${formatDate(c.created_at)}</td>
      <td>${c.retry_count}</td>
    </tr>
  `).join('');
}

function renderRecentDefects(tbody, defects) {
  if (!defects || defects.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="table-empty">No defects logged</td></tr>';
    return;
  }
  tbody.innerHTML = defects.map(d => `
    <tr>
      <td>${escapeHtml(d.title)}</td>
      <td><span class="badge badge-${d.severity}">${d.severity}</span></td>
      <td><span class="badge badge-${d.priority}">${d.priority}</span></td>
      <td><span class="badge badge-${d.status === 'open' ? 'danger' : 'success'}">${d.status}</span></td>
    </tr>
  `).join('');
}

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
