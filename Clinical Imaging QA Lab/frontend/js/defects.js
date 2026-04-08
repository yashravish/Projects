/**
 * Defects page logic — handles defect form submission
 * and renders the defect list table.
 */
document.addEventListener('DOMContentLoaded', () => {
  loadDefects();

  const form = document.getElementById('defect-form');
  form.addEventListener('submit', submitDefect);
});

async function loadDefects() {
  const tbody = document.getElementById('defects-body');
  const alertEl = document.getElementById('defects-alert');

  try {
    const defects = await API.getDefects();
    alertEl.hidden = true;
    renderDefects(tbody, defects);
  } catch (err) {
    alertEl.textContent = 'Failed to load defects: ' + err.message;
    alertEl.className = 'alert alert-danger';
    alertEl.hidden = false;
  }
}

async function submitDefect(e) {
  e.preventDefault();
  clearErrors();

  const title = document.getElementById('defect-title').value.trim();
  const severity = document.getElementById('defect-severity').value;
  const priority = document.getElementById('defect-priority').value;
  const environment = document.getElementById('defect-environment').value.trim();
  const steps = document.getElementById('defect-steps').value.trim();
  const expected = document.getElementById('defect-expected').value.trim();
  const actual = document.getElementById('defect-actual').value.trim();

  let valid = true;
  if (!title) { showFieldError('defect-title', 'Title is required'); valid = false; }
  if (!severity) { showFieldError('defect-severity', 'Severity is required'); valid = false; }
  if (!priority) { showFieldError('defect-priority', 'Priority is required'); valid = false; }

  if (!valid) return;

  const submitBtn = document.getElementById('defect-submit');
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Saving…';

  const alertEl = document.getElementById('defects-alert');

  try {
    await API.createDefect({
      title,
      severity,
      priority,
      environment: environment || null,
      steps_to_reproduce: steps || null,
      expected_result: expected || null,
      actual_result: actual || null,
    });

    alertEl.textContent = 'Defect logged successfully!';
    alertEl.className = 'alert alert-success';
    alertEl.hidden = false;
    alertEl.setAttribute('role', 'status');

    document.getElementById('defect-form').reset();
    loadDefects();
  } catch (err) {
    alertEl.textContent = 'Failed to log defect: ' + err.message;
    alertEl.className = 'alert alert-danger';
    alertEl.hidden = false;
    alertEl.setAttribute('role', 'alert');
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = 'Submit Defect';
  }
}

function renderDefects(tbody, defects) {
  if (!defects || defects.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No defects logged yet</td></tr>';
    return;
  }
  tbody.innerHTML = defects.map(d => `
    <tr>
      <td>${d.id}</td>
      <td>${escapeHtml(d.title)}</td>
      <td><span class="badge badge-${d.severity}">${d.severity}</span></td>
      <td><span class="badge badge-${d.priority}">${d.priority}</span></td>
      <td><span class="badge badge-${d.status === 'open' ? 'danger' : 'success'}">${d.status}</span></td>
      <td>${formatDate(d.created_at)}</td>
    </tr>
  `).join('');
}

function showFieldError(fieldId, message) {
  const input = document.getElementById(fieldId);
  const errorSpan = document.getElementById(fieldId + '-error');
  input.classList.add('error');
  input.setAttribute('aria-invalid', 'true');
  if (errorSpan) {
    errorSpan.textContent = message;
    errorSpan.classList.add('visible');
  }
}

function clearErrors() {
  document.querySelectorAll('.form-input, .form-select, .form-textarea').forEach(el => {
    el.classList.remove('error');
    el.removeAttribute('aria-invalid');
  });
  document.querySelectorAll('.form-error').forEach(el => {
    el.classList.remove('visible');
    el.textContent = '';
  });
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
