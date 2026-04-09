<?php
/**
 * Reports Page
 * Clinical Image Intake Portal
 *
 * Admin reporting dashboard with case counts by status, priority,
 * imaging type, recent escalations, and failed sync summary.
 */

require_once __DIR__ . '/../includes/auth.php';
requireLogin();

require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/CaseService.php';
require_once __DIR__ . '/../services/IntegrationService.php';

$caseService    = new CaseService();
$integrationSvc = new IntegrationService();

// ── Gather Report Data ──────────────────────────────────────
$openCases       = $caseService->getOpenCasesCount();
$byStatus        = $caseService->getCountByStatus();
$byImagingType   = $caseService->getCountByImagingType();
$byPriority      = $caseService->getCountByPriority();
$escalations     = $caseService->getRecentEscalations(5);
$failedSyncs     = $integrationSvc->getFailedSyncs(5);
$failedSyncCount = $integrationSvc->getFailedSyncCount();

$pageTitle = 'Reports';
include __DIR__ . '/../includes/header.php';
?>

<div class="page-header">
    <div>
        <h1>Reports</h1>
        <p class="page-subtitle">Case metrics and operational summary</p>
    </div>
</div>

<!-- Summary Cards -->
<div class="stats-grid">
    <div class="stat-card" id="statOpen">
        <div class="stat-value"><?= $openCases ?></div>
        <div class="stat-label">Open Cases</div>
    </div>
    <div class="stat-card stat-escalated" id="statEscalated">
        <div class="stat-value"><?= count($escalations) ?></div>
        <div class="stat-label">Active Escalations</div>
    </div>
    <div class="stat-card stat-failed" id="statFailed">
        <div class="stat-value"><?= $failedSyncCount ?></div>
        <div class="stat-label">Failed Syncs</div>
    </div>
</div>

<div class="report-grid">

    <!-- Cases by Status -->
    <div class="card" id="reportByStatus">
        <div class="card-header">
            <h2>Cases by Status</h2>
        </div>
        <div class="card-body">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Count</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
                    <?php
                    $totalCases = array_sum(array_column($byStatus, 'count'));
                    foreach ($byStatus as $row):
                        $pct = $totalCases > 0 ? round(($row['count'] / $totalCases) * 100) : 0;
                    ?>
                    <tr>
                        <td><span class="badge <?= statusBadgeClass($row['status']) ?>"><?= h($row['status']) ?></span></td>
                        <td><?= $row['count'] ?></td>
                        <td>
                            <div class="bar-container">
                                <div class="bar-fill <?= statusBadgeClass($row['status']) ?>" style="width:<?= $pct ?>%"></div>
                            </div>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Cases by Imaging Type -->
    <div class="card" id="reportByType">
        <div class="card-header">
            <h2>Cases by Imaging Type</h2>
        </div>
        <div class="card-body">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Imaging Type</th>
                        <th>Count</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($byImagingType as $row):
                        $pct = $totalCases > 0 ? round(($row['count'] / $totalCases) * 100) : 0;
                    ?>
                    <tr>
                        <td><?= h($row['imaging_type']) ?></td>
                        <td><?= $row['count'] ?></td>
                        <td>
                            <div class="bar-container">
                                <div class="bar-fill badge-review" style="width:<?= $pct ?>%"></div>
                            </div>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Cases by Priority -->
    <div class="card" id="reportByPriority">
        <div class="card-header">
            <h2>Cases by Priority</h2>
        </div>
        <div class="card-body">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Count</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($byPriority as $row):
                        $pct = $totalCases > 0 ? round(($row['count'] / $totalCases) * 100) : 0;
                    ?>
                    <tr>
                        <td><span class="badge <?= priorityBadgeClass($row['priority']) ?>"><?= h($row['priority']) ?></span></td>
                        <td><?= $row['count'] ?></td>
                        <td>
                            <div class="bar-container">
                                <div class="bar-fill <?= priorityBadgeClass($row['priority']) ?>" style="width:<?= $pct ?>%"></div>
                            </div>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Recent Escalations -->
    <div class="card" id="reportEscalations">
        <div class="card-header">
            <h2>Recent Escalations</h2>
        </div>
        <div class="card-body">
            <?php if (empty($escalations)): ?>
                <p class="empty-state-text">No active escalations.</p>
            <?php else: ?>
                <table class="data-table compact-table">
                    <thead>
                        <tr>
                            <th>Case</th>
                            <th>Patient</th>
                            <th>Clinic</th>
                            <th>Priority</th>
                            <th>Updated</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($escalations as $e): ?>
                        <tr>
                            <td><a href="<?= BASE_URL ?>pages/case-detail.php?id=<?= $e['id'] ?>">#<?= $e['id'] ?></a></td>
                            <td><?= h($e['patient_last_name'] . ', ' . $e['patient_first_name']) ?></td>
                            <td><?= h($e['clinic_name']) ?></td>
                            <td><span class="badge <?= priorityBadgeClass($e['priority']) ?>"><?= h($e['priority']) ?></span></td>
                            <td><?= formatDateTime($e['updated_at']) ?></td>
                        </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            <?php endif; ?>
        </div>
    </div>

    <!-- Failed Syncs -->
    <div class="card" id="reportFailedSyncs">
        <div class="card-header">
            <h2>Failed Sync Attempts</h2>
        </div>
        <div class="card-body">
            <?php if (empty($failedSyncs)): ?>
                <p class="empty-state-text">No failed sync attempts.</p>
            <?php else: ?>
                <table class="data-table compact-table">
                    <thead>
                        <tr>
                            <th>Case</th>
                            <th>Error</th>
                            <th>By</th>
                            <th>When</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($failedSyncs as $fs): ?>
                        <tr>
                            <td><a href="<?= BASE_URL ?>pages/case-detail.php?id=<?= $fs['case_id'] ?>">#<?= $fs['case_id'] ?></a></td>
                            <td class="truncate-text"><?= h($fs['error_message'] ?: 'Unknown error') ?></td>
                            <td><?= h($fs['attempted_by_name'] ?? '—') ?></td>
                            <td><?= formatDateTime($fs['attempted_at']) ?></td>
                        </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            <?php endif; ?>
        </div>
    </div>

</div>

<?php include __DIR__ . '/../includes/footer.php'; ?>
