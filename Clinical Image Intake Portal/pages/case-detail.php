<?php
/**
 * Case Detail Page
 * Clinical Image Intake Portal
 *
 * Displays full case information, status history, support notes,
 * SOAP verification, and REST sync status. Supports AJAX actions.
 */

require_once __DIR__ . '/../includes/auth.php';
requireLogin();

require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/CaseService.php';
require_once __DIR__ . '/../services/IntegrationService.php';
require_once __DIR__ . '/../services/SoapVerificationService.php';

// ── Get Case ─────────────────────────────────────────────────
$caseId = (int) ($_GET['id'] ?? 0);
if ($caseId <= 0) {
    setFlash('danger', 'Invalid case ID.');
    redirect('pages/dashboard.php');
}

$caseService       = new CaseService();
$integrationSvc    = new IntegrationService();
$soapSvc           = new SoapVerificationService();

$case = $caseService->getCaseById($caseId);
if (!$case) {
    setFlash('danger', 'Case not found.');
    redirect('pages/dashboard.php');
}

$statusHistory    = $caseService->getCaseStatusHistory($caseId);
$supportNotes     = $caseService->getSupportNotes($caseId);
$integrationLogs  = $integrationSvc->getIntegrationLogs($caseId);
$soapVerification = $soapSvc->getVerification($caseId);

$pageTitle = "Case #{$caseId} — " . h($case['patient_last_name']);
$pageScripts = ['case-detail.js'];
include __DIR__ . '/../includes/header.php';
?>

<div class="page-header">
    <div>
        <h1>Case #<?= $case['id'] ?></h1>
        <p class="page-subtitle"><?= h($case['patient_first_name'] . ' ' . $case['patient_last_name']) ?> — <?= h($case['clinic_name']) ?></p>
    </div>
    <a href="<?= BASE_URL ?>pages/dashboard.php" class="btn btn-outline" id="btnBackDashboard">← Dashboard</a>
</div>

<div class="detail-grid">

    <!-- Left Column: Case Information -->
    <div class="detail-main">

        <!-- Case Summary Card -->
        <div class="card" id="caseSummaryCard">
            <div class="card-header">
                <h2>Case Summary</h2>
                <div class="badge-group">
                    <span class="badge <?= statusBadgeClass($case['status']) ?>" id="caseStatusBadge"><?= h($case['status']) ?></span>
                    <span class="badge <?= priorityBadgeClass($case['priority']) ?>"><?= h($case['priority']) ?> Priority</span>
                </div>
            </div>
            <div class="card-body">
                <div class="detail-fields">
                    <div class="detail-field">
                        <label>Patient Name</label>
                        <span><?= h($case['patient_first_name'] . ' ' . $case['patient_last_name']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Date of Birth</label>
                        <span><?= formatDate($case['date_of_birth']) ?> (Age: <?= calculateAge($case['date_of_birth']) ?>)</span>
                    </div>
                    <div class="detail-field">
                        <label>Clinic</label>
                        <span><?= h($case['clinic_name']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Provider</label>
                        <span><?= h($case['provider_name']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Imaging Type</label>
                        <span><?= h($case['imaging_type']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Body Area</label>
                        <span><?= h($case['body_area'] ?: '—') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Email</label>
                        <span><?= h($case['patient_email'] ?: '—') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Phone</label>
                        <span><?= h($case['patient_phone'] ?: '—') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Insurance ID</label>
                        <span><?= h($case['insurance_id'] ?: '—') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Image File</label>
                        <span><?= h($case['image_filename'] ?: '—') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Assigned To</label>
                        <span><?= h($case['assigned_name'] ?: 'Unassigned') ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Created</label>
                        <span><?= formatDateTime($case['created_at']) ?></span>
                    </div>
                </div>
                <?php if (!empty($case['symptoms_notes'])): ?>
                <div class="symptoms-section">
                    <label>Symptoms / Clinical Notes</label>
                    <div class="symptoms-text"><?= nl2br(h($case['symptoms_notes'])) ?></div>
                </div>
                <?php endif; ?>
            </div>
        </div>

        <!-- Status History Card -->
        <div class="card" id="statusHistoryCard">
            <div class="card-header">
                <h2>Status History</h2>
            </div>
            <div class="card-body">
                <?php if (empty($statusHistory)): ?>
                    <p class="empty-state-text">No status changes recorded.</p>
                <?php else: ?>
                    <div class="timeline">
                        <?php foreach ($statusHistory as $h): ?>
                        <div class="timeline-item">
                            <div class="timeline-marker"></div>
                            <div class="timeline-content">
                                <div class="timeline-header">
                                    <?php if ($h['old_status']): ?>
                                        <span class="badge badge-sm <?= statusBadgeClass($h['old_status']) ?>"><?= h($h['old_status']) ?></span>
                                        <span class="timeline-arrow">→</span>
                                    <?php endif; ?>
                                    <span class="badge badge-sm <?= statusBadgeClass($h['new_status']) ?>"><?= h($h['new_status']) ?></span>
                                </div>
                                <div class="timeline-meta">
                                    <?= h($h['changed_by_name'] ?? 'System') ?> — <?= formatDateTime($h['changed_at']) ?>
                                </div>
                                <?php if (!empty($h['notes'])): ?>
                                    <div class="timeline-notes"><?= h($h['notes']) ?></div>
                                <?php endif; ?>
                            </div>
                        </div>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>
            </div>
        </div>

        <!-- Support Notes Card -->
        <div class="card" id="supportNotesCard">
            <div class="card-header">
                <h2>Support Notes</h2>
            </div>
            <div class="card-body">
                <!-- Add Note Form -->
                <div class="add-note-form" id="addNoteForm">
                    <div class="form-row">
                        <div class="form-group" style="flex:2">
                            <textarea id="noteBody" class="form-control" rows="2" placeholder="Add a support or troubleshooting note..."></textarea>
                        </div>
                        <div class="form-group">
                            <select id="noteType" class="form-control">
                                <?php foreach (getNoteTypes() as $nt): ?>
                                    <option value="<?= h($nt) ?>"><?= h(noteTypeLabel($nt)) ?></option>
                                <?php endforeach; ?>
                            </select>
                        </div>
                    </div>
                    <button class="btn btn-primary btn-sm" id="btnAddNote" data-case-id="<?= $case['id'] ?>">Add Note</button>
                </div>

                <!-- Notes List -->
                <div class="notes-list" id="notesList">
                    <?php if (empty($supportNotes)): ?>
                        <p class="empty-state-text" id="noNotesMsg">No support notes yet.</p>
                    <?php else: ?>
                        <?php foreach ($supportNotes as $note): ?>
                        <div class="note-item">
                            <div class="note-header">
                                <span class="note-author"><?= h($note['author_name']) ?></span>
                                <span class="note-type-badge badge-<?= h($note['note_type']) ?>"><?= h(noteTypeLabel($note['note_type'])) ?></span>
                                <span class="note-date"><?= formatDateTime($note['created_at']) ?></span>
                            </div>
                            <div class="note-body"><?= nl2br(h($note['note_body'])) ?></div>
                        </div>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </div>
            </div>
        </div>
    </div>

    <!-- Right Column: Actions & Integrations -->
    <div class="detail-sidebar">

        <!-- Status Update Card -->
        <?php if ($case['status'] !== 'Closed'): ?>
        <div class="card" id="statusUpdateCard">
            <div class="card-header">
                <h3>Update Status</h3>
            </div>
            <div class="card-body">
                <div class="form-group">
                    <select id="detailNewStatus" class="form-control">
                        <?php foreach (getStatusOptions() as $s): ?>
                            <option value="<?= h($s) ?>" <?= $case['status'] === $s ? 'selected' : '' ?>><?= h($s) ?></option>
                        <?php endforeach; ?>
                    </select>
                </div>
                <div class="form-group">
                    <textarea id="detailStatusNotes" class="form-control" rows="2" placeholder="Notes (optional)"></textarea>
                </div>
                <button class="btn btn-primary btn-block" id="btnUpdateStatus" data-case-id="<?= $case['id'] ?>">
                    Update Status
                </button>
            </div>
        </div>
        <?php endif; ?>

        <!-- REST Sync Card -->
        <div class="card" id="syncCard">
            <div class="card-header">
                <h3>External Sync</h3>
            </div>
            <div class="card-body">
                <div class="detail-field">
                    <label>Sync Status</label>
                    <span class="badge <?= syncBadgeClass($case['external_sync_status']) ?>" id="syncStatusBadge">
                        <?= h(ucwords(str_replace('_', ' ', $case['external_sync_status']))) ?>
                    </span>
                </div>
                <?php if ($case['external_reference_id']): ?>
                <div class="detail-field">
                    <label>External Ref ID</label>
                    <span class="mono-text"><?= h($case['external_reference_id']) ?></span>
                </div>
                <?php endif; ?>
                <button class="btn btn-primary btn-block" id="btnSyncCase" data-case-id="<?= $case['id'] ?>">
                    <?= $case['external_sync_status'] === 'failed' ? '↻ Retry Sync' : '⇄ Sync Now' ?>
                </button>

                <?php if (!empty($integrationLogs)): ?>
                <div class="integration-log-summary">
                    <h4>Recent Sync Attempts</h4>
                    <?php foreach (array_slice($integrationLogs, 0, 3) as $log): ?>
                    <div class="log-entry <?= $log['success'] ? 'log-success' : 'log-failure' ?>">
                        <span class="log-status"><?= $log['success'] ? '✓' : '✗' ?></span>
                        <span class="log-date"><?= formatDateTime($log['attempted_at']) ?></span>
                        <?php if (!$log['success'] && $log['error_message']): ?>
                            <span class="log-error"><?= h($log['error_message']) ?></span>
                        <?php endif; ?>
                    </div>
                    <?php endforeach; ?>
                </div>
                <?php endif; ?>
            </div>
        </div>

        <!-- SOAP Verification Card -->
        <div class="card" id="soapCard">
            <div class="card-header">
                <h3>Insurance Verification</h3>
            </div>
            <div class="card-body">
                <?php if ($soapVerification): ?>
                <div class="verification-result">
                    <div class="detail-field">
                        <label>Status</label>
                        <span class="badge <?= $soapVerification['clinic_approved'] ? 'badge-verified' : 'badge-escalated' ?>">
                            <?= h($soapVerification['verification_status']) ?>
                        </span>
                    </div>
                    <div class="detail-field">
                        <label>Policy Type</label>
                        <span><?= h($soapVerification['policy_type']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Message</label>
                        <span><?= h($soapVerification['message']) ?></span>
                    </div>
                    <div class="detail-field">
                        <label>Checked At</label>
                        <span><?= formatDateTime($soapVerification['checked_at']) ?></span>
                    </div>
                </div>
                <?php else: ?>
                    <p class="empty-state-text">No verification on file.</p>
                <?php endif; ?>

                <button class="btn btn-primary btn-block" id="btnVerifySoap"
                        data-case-id="<?= $case['id'] ?>"
                        data-clinic="<?= h($case['clinic_name']) ?>"
                        data-insurance="<?= h($case['insurance_id']) ?>">
                    <?= $soapVerification ? '↻ Re-Verify' : '✓ Verify Coverage' ?>
                </button>
                <div id="soapResultArea"></div>
            </div>
        </div>

    </div>
</div>

<?php include __DIR__ . '/../includes/footer.php'; ?>
