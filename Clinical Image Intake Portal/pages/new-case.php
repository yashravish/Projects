<?php
/**
 * New Case Intake Form
 * Clinical Image Intake Portal
 *
 * Server-rendered intake form with client-side and server-side validation.
 * Mimics a real clinical wireframe/specification form.
 */

require_once __DIR__ . '/../includes/auth.php';
requireLogin();

require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/CaseService.php';

$caseService = new CaseService();
$errors = [];
$formData = [];

// ── Handle Form Submission ──────────────────────────────────
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_once __DIR__ . '/../includes/csrf.php';
    requireCsrf();

    // Collect & sanitize inputs
    $formData = [
        'patient_first_name' => sanitizeInput($_POST['patient_first_name'] ?? ''),
        'patient_last_name'  => sanitizeInput($_POST['patient_last_name'] ?? ''),
        'date_of_birth'      => sanitizeInput($_POST['date_of_birth'] ?? ''),
        'clinic_name'        => sanitizeInput($_POST['clinic_name'] ?? ''),
        'provider_name'      => sanitizeInput($_POST['provider_name'] ?? ''),
        'imaging_type'       => sanitizeInput($_POST['imaging_type'] ?? ''),
        'body_area'          => sanitizeInput($_POST['body_area'] ?? ''),
        'priority'           => sanitizeInput($_POST['priority'] ?? 'Medium'),
        'symptoms_notes'     => sanitizeInput($_POST['symptoms_notes'] ?? ''),
        'image_filename'     => sanitizeInput($_POST['image_filename'] ?? ''),
        'insurance_id'       => sanitizeInput($_POST['insurance_id'] ?? ''),
        'patient_email'      => sanitizeInput($_POST['patient_email'] ?? ''),
        'patient_phone'      => sanitizeInput($_POST['patient_phone'] ?? ''),
        'assigned_to'        => (int) ($_POST['assigned_to'] ?? 0),
        'created_by'         => currentUserId(),
    ];

    // ── Server-Side Validation ──────────────────────────────
    if (!isRequired($formData['patient_first_name'])) {
        $errors['patient_first_name'] = 'Patient first name is required.';
    }
    if (!isRequired($formData['patient_last_name'])) {
        $errors['patient_last_name'] = 'Patient last name is required.';
    }
    if (!isRequired($formData['date_of_birth'])) {
        $errors['date_of_birth'] = 'Date of birth is required.';
    } elseif (!isValidDate($formData['date_of_birth'])) {
        $errors['date_of_birth'] = 'Invalid date format. Use YYYY-MM-DD.';
    }
    if (!isRequired($formData['clinic_name'])) {
        $errors['clinic_name'] = 'Clinic name is required.';
    }
    if (!isRequired($formData['provider_name'])) {
        $errors['provider_name'] = 'Provider name is required.';
    }
    if (!in_array($formData['imaging_type'], getImagingTypes())) {
        $errors['imaging_type'] = 'Please select a valid imaging type.';
    }
    if (!in_array($formData['priority'], getPriorityOptions())) {
        $errors['priority'] = 'Please select a valid priority level.';
    }
    if (!empty($formData['patient_email']) && !isValidEmail($formData['patient_email'])) {
        $errors['patient_email'] = 'Invalid email address format.';
    }
    if (!empty($formData['patient_phone']) && !isValidPhone($formData['patient_phone'])) {
        $errors['patient_phone'] = 'Invalid phone number format.';
    }

    // ── Create Case if Validation Passes ────────────────────
    if (empty($errors)) {
        try {
            $caseId = $caseService->createCase($formData);
            setFlash('success', "Case #{$caseId} created successfully.");
            redirect('pages/case-detail.php?id=' . $caseId);
        } catch (Exception $e) {
            $errors['general'] = 'An error occurred while creating the case. Please try again.';
        }
    }
}

// Get user list for assignment dropdown
$users = getUserList();

$pageTitle = 'New Case Intake';
$pageScripts = ['validation.js'];
include __DIR__ . '/../includes/header.php';
?>

<div class="page-header">
    <div>
        <h1>New Case Intake</h1>
        <p class="page-subtitle">Submit a new imaging case for review</p>
    </div>
    <a href="<?= BASE_URL ?>pages/dashboard.php" class="btn btn-outline" id="btnBackDashboard">← Back to Dashboard</a>
</div>

<?php if (!empty($errors)): ?>
<div class="alert alert-danger" id="serverErrors">
    <strong>Please correct the following errors:</strong>
    <ul>
        <?php foreach ($errors as $err): ?>
            <li><?= h($err) ?></li>
        <?php endforeach; ?>
    </ul>
</div>
<?php endif; ?>

<div class="card form-card">
    <form method="POST" action="" id="intakeForm" novalidate>
        <?php csrfField(); ?>

        <!-- Patient Information Section -->
        <fieldset class="form-section">
            <legend>Patient Information</legend>
            <div class="form-row">
                <div class="form-group">
                    <label for="patient_first_name">First Name <span class="required">*</span></label>
                    <input type="text" id="patient_first_name" name="patient_first_name"
                           class="form-control <?= isset($errors['patient_first_name']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['patient_first_name'] ?? '') ?>"
                           required maxlength="100" placeholder="e.g. Maria">
                    <div class="invalid-feedback" id="err_patient_first_name"></div>
                </div>
                <div class="form-group">
                    <label for="patient_last_name">Last Name <span class="required">*</span></label>
                    <input type="text" id="patient_last_name" name="patient_last_name"
                           class="form-control <?= isset($errors['patient_last_name']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['patient_last_name'] ?? '') ?>"
                           required maxlength="100" placeholder="e.g. Santos">
                    <div class="invalid-feedback" id="err_patient_last_name"></div>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="date_of_birth">Date of Birth <span class="required">*</span></label>
                    <input type="date" id="date_of_birth" name="date_of_birth"
                           class="form-control <?= isset($errors['date_of_birth']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['date_of_birth'] ?? '') ?>"
                           required max="<?= date('Y-m-d') ?>">
                    <div class="invalid-feedback" id="err_date_of_birth"></div>
                </div>
                <div class="form-group">
                    <label for="insurance_id">Insurance ID</label>
                    <input type="text" id="insurance_id" name="insurance_id"
                           class="form-control"
                           value="<?= h($formData['insurance_id'] ?? '') ?>"
                           maxlength="50" placeholder="e.g. INS-12345">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="patient_email">Email</label>
                    <input type="email" id="patient_email" name="patient_email"
                           class="form-control <?= isset($errors['patient_email']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['patient_email'] ?? '') ?>"
                           maxlength="100" placeholder="patient@email.com">
                    <div class="invalid-feedback" id="err_patient_email"></div>
                </div>
                <div class="form-group">
                    <label for="patient_phone">Phone</label>
                    <input type="tel" id="patient_phone" name="patient_phone"
                           class="form-control <?= isset($errors['patient_phone']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['patient_phone'] ?? '') ?>"
                           maxlength="30" placeholder="555-0101">
                    <div class="invalid-feedback" id="err_patient_phone"></div>
                </div>
            </div>
        </fieldset>

        <!-- Clinical Details Section -->
        <fieldset class="form-section">
            <legend>Clinical Details</legend>
            <div class="form-row">
                <div class="form-group">
                    <label for="clinic_name">Clinic Name <span class="required">*</span></label>
                    <input type="text" id="clinic_name" name="clinic_name"
                           class="form-control <?= isset($errors['clinic_name']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['clinic_name'] ?? '') ?>"
                           required maxlength="150" placeholder="e.g. Riverside Medical Center">
                    <div class="invalid-feedback" id="err_clinic_name"></div>
                </div>
                <div class="form-group">
                    <label for="provider_name">Provider Name <span class="required">*</span></label>
                    <input type="text" id="provider_name" name="provider_name"
                           class="form-control <?= isset($errors['provider_name']) ? 'is-invalid' : '' ?>"
                           value="<?= h($formData['provider_name'] ?? '') ?>"
                           required maxlength="150" placeholder="e.g. Dr. Angela Cruz">
                    <div class="invalid-feedback" id="err_provider_name"></div>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="imaging_type">Imaging Type <span class="required">*</span></label>
                    <select id="imaging_type" name="imaging_type"
                            class="form-control <?= isset($errors['imaging_type']) ? 'is-invalid' : '' ?>" required>
                        <option value="">— Select Type —</option>
                        <?php foreach (getImagingTypes() as $type): ?>
                            <option value="<?= h($type) ?>" <?= ($formData['imaging_type'] ?? '') === $type ? 'selected' : '' ?>>
                                <?= h($type) ?>
                            </option>
                        <?php endforeach; ?>
                    </select>
                    <div class="invalid-feedback" id="err_imaging_type"></div>
                </div>
                <div class="form-group">
                    <label for="body_area">Body Area</label>
                    <input type="text" id="body_area" name="body_area"
                           class="form-control"
                           value="<?= h($formData['body_area'] ?? '') ?>"
                           maxlength="100" placeholder="e.g. Left forearm">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="priority">Priority <span class="required">*</span></label>
                    <select id="priority" name="priority" class="form-control" required>
                        <?php foreach (getPriorityOptions() as $p): ?>
                            <option value="<?= h($p) ?>" <?= ($formData['priority'] ?? 'Medium') === $p ? 'selected' : '' ?>>
                                <?= h($p) ?>
                            </option>
                        <?php endforeach; ?>
                    </select>
                </div>
                <div class="form-group">
                    <label for="assigned_to">Assign To</label>
                    <select id="assigned_to" name="assigned_to" class="form-control">
                        <option value="">— Unassigned —</option>
                        <?php foreach ($users as $uid => $uname): ?>
                            <option value="<?= $uid ?>" <?= ($formData['assigned_to'] ?? 0) == $uid ? 'selected' : '' ?>>
                                <?= h($uname) ?>
                            </option>
                        <?php endforeach; ?>
                    </select>
                </div>
            </div>
        </fieldset>

        <!-- Notes & Image Section -->
        <fieldset class="form-section">
            <legend>Notes &amp; Image</legend>
            <div class="form-group">
                <label for="symptoms_notes">Symptoms / Clinical Notes</label>
                <textarea id="symptoms_notes" name="symptoms_notes" class="form-control"
                          rows="4" placeholder="Describe symptoms, clinical observations, or relevant history..."><?= h($formData['symptoms_notes'] ?? '') ?></textarea>
            </div>
            <div class="form-group">
                <label for="image_filename">Image Filename / Path</label>
                <input type="text" id="image_filename" name="image_filename"
                       class="form-control"
                       value="<?= h($formData['image_filename'] ?? '') ?>"
                       maxlength="255" placeholder="e.g. patient_forearm_001.jpg">
                <small class="form-hint">Enter the image filename or file path reference for this case.</small>
            </div>
        </fieldset>

        <!-- Form Actions -->
        <div class="form-actions">
            <button type="submit" class="btn btn-primary" id="btnSubmitCase">Submit Case</button>
            <a href="<?= BASE_URL ?>pages/dashboard.php" class="btn btn-outline" id="btnCancelCase">Cancel</a>
        </div>
    </form>
</div>

<?php include __DIR__ . '/../includes/footer.php'; ?>
