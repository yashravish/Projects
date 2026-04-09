<?php
/**
 * Issues & Logs Page (Admin Only)
 * Clinical Image Intake Portal
 *
 * Troubleshooting page showing recent application errors, warnings,
 * and system logs. Helps admins diagnose and resolve issues.
 */

require_once __DIR__ . '/../includes/auth.php';
requireLogin();

// Only admins can view system logs
if (!isAdmin()) {
    setFlash('danger', 'Access denied. Admin privileges required.');
    redirect('pages/dashboard.php');
}

require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/LogService.php';

$logService = new LogService();

// ── Gather Log Data ─────────────────────────────────────────
$filterLevel = sanitizeInput($_GET['level'] ?? '');
$recentIssues = $logService->getRecentIssues(30);
$allLogs      = $logService->getRecentLogs(50, $filterLevel ?: null);
$logCounts    = $logService->getLogCountsByLevel();

$pageTitle = 'Issues & Logs';
include __DIR__ . '/../includes/header.php';
?>

<div class="page-header">
    <div>
        <h1>Issues &amp; Application Logs</h1>
        <p class="page-subtitle">Monitor application health and troubleshoot problems</p>
    </div>
</div>

<!-- Log Level Summary -->
<div class="stats-grid">
    <?php
    $levelCounts = [];
    foreach ($logCounts as $lc) {
        $levelCounts[$lc['level']] = $lc['count'];
    }
    ?>
    <div class="stat-card stat-critical" id="statCritical">
        <div class="stat-value"><?= $levelCounts['critical'] ?? 0 ?></div>
        <div class="stat-label">Critical</div>
    </div>
    <div class="stat-card stat-failed" id="statErrors">
        <div class="stat-value"><?= $levelCounts['error'] ?? 0 ?></div>
        <div class="stat-label">Errors</div>
    </div>
    <div class="stat-card stat-warning" id="statWarnings">
        <div class="stat-value"><?= $levelCounts['warning'] ?? 0 ?></div>
        <div class="stat-label">Warnings</div>
    </div>
    <div class="stat-card" id="statInfo">
        <div class="stat-value"><?= $levelCounts['info'] ?? 0 ?></div>
        <div class="stat-label">Info</div>
    </div>
</div>

<!-- Filter by Level -->
<div class="card filter-card">
    <form method="GET" action="" class="filter-form" id="logFilterForm">
        <div class="filter-group">
            <label for="logLevelFilter">Filter by Level</label>
            <select id="logLevelFilter" name="level">
                <option value="">All Levels</option>
                <option value="critical" <?= $filterLevel === 'critical' ? 'selected' : '' ?>>Critical</option>
                <option value="error" <?= $filterLevel === 'error' ? 'selected' : '' ?>>Error</option>
                <option value="warning" <?= $filterLevel === 'warning' ? 'selected' : '' ?>>Warning</option>
                <option value="info" <?= $filterLevel === 'info' ? 'selected' : '' ?>>Info</option>
            </select>
        </div>
        <div class="filter-actions">
            <button type="submit" class="btn btn-primary btn-sm" id="btnApplyLogFilter">Apply</button>
            <a href="<?= BASE_URL ?>pages/issues.php" class="btn btn-outline btn-sm">Clear</a>
        </div>
    </form>
</div>

<!-- Recent Issues (Errors & Warnings) -->
<div class="card" id="recentIssuesCard">
    <div class="card-header">
        <h2>Recent Issues</h2>
    </div>
    <div class="card-body">
        <?php if (empty($recentIssues)): ?>
            <p class="empty-state-text">No recent issues. System is healthy.</p>
        <?php else: ?>
            <div class="table-responsive">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Source</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($recentIssues as $issue): ?>
                    <tr class="log-row-<?= h($issue['level']) ?>">
                        <td>
                            <span class="log-level-badge log-<?= h($issue['level']) ?>">
                                <?= h(strtoupper($issue['level'])) ?>
                            </span>
                        </td>
                        <td><?= h($issue['message']) ?></td>
                        <td class="mono-text"><?= h($issue['file'] ? basename($issue['file']) . ':' . $issue['line'] : '—') ?></td>
                        <td><?= formatDateTime($issue['created_at']) ?></td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
            </div>
        <?php endif; ?>
    </div>
</div>

<!-- All Logs -->
<div class="card" id="allLogsCard">
    <div class="card-header">
        <h2>Application Log <?= $filterLevel ? '(' . ucfirst($filterLevel) . ')' : '' ?></h2>
    </div>
    <div class="card-body">
        <?php if (empty($allLogs)): ?>
            <p class="empty-state-text">No log entries found.</p>
        <?php else: ?>
            <div class="table-responsive">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Context</th>
                        <th>Source</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($allLogs as $log): ?>
                    <tr class="log-row-<?= h($log['level']) ?>">
                        <td><?= $log['id'] ?></td>
                        <td>
                            <span class="log-level-badge log-<?= h($log['level']) ?>">
                                <?= h(strtoupper($log['level'])) ?>
                            </span>
                        </td>
                        <td><?= h($log['message']) ?></td>
                        <td class="mono-text truncate-text"><?= h($log['context'] ?: '—') ?></td>
                        <td class="mono-text"><?= h($log['file'] ? basename($log['file']) . ':' . $log['line'] : '—') ?></td>
                        <td><?= formatDateTime($log['created_at']) ?></td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
            </div>
        <?php endif; ?>
    </div>
</div>

<?php include __DIR__ . '/../includes/footer.php'; ?>
