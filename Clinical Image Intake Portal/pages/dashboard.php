<?php
/**
 * Dashboard Page
 * Clinical Image Intake Portal
 *
 * Main case listing with search/filter, pagination, sortable display,
 * and inline AJAX status updates.
 */

require_once __DIR__ . '/../includes/auth.php';
requireLogin();

require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/CaseService.php';

$caseService = new CaseService();

// ── Gather filter parameters ────────────────────────────────
$filters = [
    'search'       => sanitizeInput($_GET['search'] ?? ''),
    'status'       => sanitizeInput($_GET['status'] ?? ''),
    'priority'     => sanitizeInput($_GET['priority'] ?? ''),
    'imaging_type' => sanitizeInput($_GET['imaging_type'] ?? ''),
];

$page = max(1, (int) ($_GET['page'] ?? 1));

// ── Fetch cases and total count ─────────────────────────────
$cases      = $caseService->getAllCases($filters, $page);
$totalCases = $caseService->getTotalCasesCount($filters);
$totalPages = max(1, (int) ceil($totalCases / CASES_PER_PAGE));

$pageTitle = 'Dashboard';
$pageScripts = ['dashboard.js'];
include __DIR__ . '/../includes/header.php';
?>

<div class="page-header">
    <div>
        <h1>Case Dashboard</h1>
        <p class="page-subtitle">Manage and review imaging intake cases</p>
    </div>
    <a href="<?= BASE_URL ?>pages/new-case.php" class="btn btn-primary" id="btnNewCase">
        + New Case
    </a>
</div>

<!-- Filters -->
<div class="card filter-card" id="filterCard">
    <form method="GET" action="" class="filter-form" id="filterForm">
        <div class="filter-group">
            <label for="filterSearch">Search</label>
            <input type="text" id="filterSearch" name="search" placeholder="Patient name or clinic..."
                   value="<?= h($filters['search']) ?>">
        </div>
        <div class="filter-group">
            <label for="filterStatus">Status</label>
            <select id="filterStatus" name="status">
                <option value="">All Statuses</option>
                <?php foreach (getStatusOptions() as $s): ?>
                    <option value="<?= h($s) ?>" <?= $filters['status'] === $s ? 'selected' : '' ?>><?= h($s) ?></option>
                <?php endforeach; ?>
            </select>
        </div>
        <div class="filter-group">
            <label for="filterPriority">Priority</label>
            <select id="filterPriority" name="priority">
                <option value="">All Priorities</option>
                <?php foreach (getPriorityOptions() as $p): ?>
                    <option value="<?= h($p) ?>" <?= $filters['priority'] === $p ? 'selected' : '' ?>><?= h($p) ?></option>
                <?php endforeach; ?>
            </select>
        </div>
        <div class="filter-group">
            <label for="filterImaging">Imaging Type</label>
            <select id="filterImaging" name="imaging_type">
                <option value="">All Types</option>
                <?php foreach (getImagingTypes() as $t): ?>
                    <option value="<?= h($t) ?>" <?= $filters['imaging_type'] === $t ? 'selected' : '' ?>><?= h($t) ?></option>
                <?php endforeach; ?>
            </select>
        </div>
        <div class="filter-actions">
            <button type="submit" class="btn btn-primary btn-sm" id="btnApplyFilters">Apply</button>
            <a href="<?= BASE_URL ?>pages/dashboard.php" class="btn btn-outline btn-sm" id="btnClearFilters">Clear</a>
        </div>
    </form>
</div>

<!-- Results Summary -->
<div class="results-summary">
    Showing <?= count($cases) ?> of <?= $totalCases ?> case<?= $totalCases !== 1 ? 's' : '' ?>
    <?php if ($page > 1): ?> — Page <?= $page ?> of <?= $totalPages ?><?php endif; ?>
</div>

<!-- Cases Table -->
<div class="card table-card">
    <div class="table-responsive">
        <table class="data-table" id="casesTable">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Patient Name</th>
                    <th>DOB</th>
                    <th>Clinic</th>
                    <th>Imaging Type</th>
                    <th>Priority</th>
                    <th>Status</th>
                    <th>Assigned To</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <?php if (empty($cases)): ?>
                <tr>
                    <td colspan="10" class="empty-state">No cases found matching your criteria.</td>
                </tr>
                <?php else: ?>
                    <?php foreach ($cases as $case): ?>
                    <tr id="caseRow-<?= $case['id'] ?>">
                        <td class="case-id">#<?= $case['id'] ?></td>
                        <td>
                            <a href="<?= BASE_URL ?>pages/case-detail.php?id=<?= $case['id'] ?>" class="patient-link">
                                <?= h($case['patient_last_name']) ?>, <?= h($case['patient_first_name']) ?>
                            </a>
                        </td>
                        <td><?= formatDate($case['date_of_birth']) ?></td>
                        <td><?= h($case['clinic_name']) ?></td>
                        <td><?= h($case['imaging_type']) ?></td>
                        <td>
                            <span class="badge <?= priorityBadgeClass($case['priority']) ?>">
                                <?= h($case['priority']) ?>
                            </span>
                        </td>
                        <td>
                            <span class="badge <?= statusBadgeClass($case['status']) ?>" id="statusBadge-<?= $case['id'] ?>">
                                <?= h($case['status']) ?>
                            </span>
                        </td>
                        <td><?= h($case['assigned_name'] ?? '—') ?></td>
                        <td><?= formatDate($case['created_at']) ?></td>
                        <td class="actions-cell">
                            <a href="<?= BASE_URL ?>pages/case-detail.php?id=<?= $case['id'] ?>" class="btn btn-xs btn-outline" title="View Details">View</a>
                            <?php if ($case['status'] !== 'Closed'): ?>
                            <button class="btn btn-xs btn-primary status-update-btn"
                                    data-case-id="<?= $case['id'] ?>"
                                    data-current-status="<?= h($case['status']) ?>"
                                    title="Update Status">Status</button>
                            <?php endif; ?>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>
    </div>
</div>

<!-- Pagination -->
<?php if ($totalPages > 1): ?>
<div class="pagination" id="pagination">
    <?php
    // Build base query string without page
    $queryParams = $filters;
    $queryParams = array_filter($queryParams);
    ?>

    <?php if ($page > 1): ?>
        <a href="?<?= http_build_query(array_merge($queryParams, ['page' => $page - 1])) ?>" class="pagination-link">← Prev</a>
    <?php endif; ?>

    <?php for ($i = 1; $i <= $totalPages; $i++): ?>
        <?php if ($i === $page): ?>
            <span class="pagination-link active"><?= $i ?></span>
        <?php else: ?>
            <a href="?<?= http_build_query(array_merge($queryParams, ['page' => $i])) ?>" class="pagination-link"><?= $i ?></a>
        <?php endif; ?>
    <?php endfor; ?>

    <?php if ($page < $totalPages): ?>
        <a href="?<?= http_build_query(array_merge($queryParams, ['page' => $page + 1])) ?>" class="pagination-link">Next →</a>
    <?php endif; ?>
</div>
<?php endif; ?>

<!-- Status Update Modal -->
<div class="modal-overlay" id="statusModal" style="display:none;">
    <div class="modal">
        <div class="modal-header">
            <h3>Update Case Status</h3>
            <button class="modal-close" id="statusModalClose">×</button>
        </div>
        <div class="modal-body">
            <p>Case: <strong id="modalCaseLabel"></strong></p>
            <div class="form-group">
                <label for="modalNewStatus">New Status</label>
                <select id="modalNewStatus" class="form-control">
                    <?php foreach (getStatusOptions() as $s): ?>
                        <option value="<?= h($s) ?>"><?= h($s) ?></option>
                    <?php endforeach; ?>
                </select>
            </div>
            <div class="form-group">
                <label for="modalStatusNotes">Notes (optional)</label>
                <textarea id="modalStatusNotes" class="form-control" rows="2" placeholder="Reason for status change..."></textarea>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-outline" id="statusModalCancel">Cancel</button>
            <button class="btn btn-primary" id="statusModalSubmit">Update Status</button>
        </div>
        <input type="hidden" id="modalCaseId" value="">
    </div>
</div>

<?php include __DIR__ . '/../includes/footer.php'; ?>
