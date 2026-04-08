/**
 * Capture page logic — handles form validation, submission,
 * and result display for imaging capture requests.
 */
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('capture-form');
  const resultArea = document.getElementById('capture-result');
  const submitBtn = document.getElementById('capture-submit');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearErrors();

    const patientId = document.getElementById('patient-id').value.trim();
    const sessionId = document.getElementById('session-id').value.trim();
    const imageType = document.getElementById('image-type').value;

    let valid = true;

    if (!patientId) {
      showFieldError('patient-id', 'Patient ID is required');
      valid = false;
    }
    if (!sessionId) {
      showFieldError('session-id', 'Session ID is required');
      valid = false;
    }
    if (!imageType) {
      showFieldError('image-type', 'Please select an image type');
      valid = false;
    }

    if (!valid) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Capturing…';

    try {
      const capture = await API.createCapture({
        patient_id: patientId,
        session_id: sessionId,
        image_type: imageType,
      });

      if (capture.capture_status === 'success') {
        resultArea.innerHTML = `
          <div class="alert alert-success" role="status" aria-live="polite">
            <strong>Capture Successful!</strong> ID: ${capture.id} — File: ${capture.file_path || 'N/A'}
          </div>
        `;
      } else {
        resultArea.innerHTML = `
          <div class="alert alert-danger" role="alert" aria-live="assertive">
            <strong>Capture Failed.</strong> ${escapeHtml(capture.error_message || 'Unknown error')}
            <br><button class="btn btn-sm btn-secondary mt-16" onclick="retryCapture(${capture.id})">Retry</button>
          </div>
        `;
      }
    } catch (err) {
      resultArea.innerHTML = `
        <div class="alert alert-danger" role="alert" aria-live="assertive">
          <strong>Error:</strong> ${escapeHtml(err.message)}
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = 'Start Capture';
    }
  });
});

async function retryCapture(captureId) {
  const resultArea = document.getElementById('capture-result');
  resultArea.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> Retrying capture…</div>';

  try {
    const capture = await API.retryCapture(captureId);
    if (capture.capture_status === 'success') {
      resultArea.innerHTML = `
        <div class="alert alert-success" role="status" aria-live="polite">
          <strong>Retry Successful!</strong> ID: ${capture.id} — Attempt #${capture.retry_count + 1}
        </div>
      `;
    } else {
      resultArea.innerHTML = `
        <div class="alert alert-danger" role="alert" aria-live="assertive">
          <strong>Retry Failed.</strong> ${escapeHtml(capture.error_message || 'Unknown error')} (Attempt #${capture.retry_count})
          <br><button class="btn btn-sm btn-secondary mt-16" onclick="retryCapture(${capture.id})">Retry Again</button>
        </div>
      `;
    }
  } catch (err) {
    resultArea.innerHTML = `
      <div class="alert alert-danger" role="alert">${escapeHtml(err.message)}</div>
    `;
  }
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
  document.querySelectorAll('.form-input, .form-select').forEach(el => {
    el.classList.remove('error');
    el.removeAttribute('aria-invalid');
  });
  document.querySelectorAll('.form-error').forEach(el => {
    el.classList.remove('visible');
    el.textContent = '';
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
