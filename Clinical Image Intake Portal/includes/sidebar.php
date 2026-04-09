<?php
/**
 * Sidebar Navigation Partial
 * Clinical Image Intake Portal
 *
 * Rendered inside the app-container for authenticated users.
 * Highlights the active page based on the current script name.
 */

$currentPage = basename($_SERVER['SCRIPT_NAME']);
?>
<aside class="sidebar" id="sidebar">
    <nav class="sidebar-nav">
        <ul>
            <li>
                <a href="<?= BASE_URL ?>pages/dashboard.php"
                   class="sidebar-link <?= $currentPage === 'dashboard.php' ? 'active' : '' ?>"
                   id="navDashboard">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
                    <span>Dashboard</span>
                </a>
            </li>
            <li>
                <a href="<?= BASE_URL ?>pages/new-case.php"
                   class="sidebar-link <?= $currentPage === 'new-case.php' ? 'active' : '' ?>"
                   id="navNewCase">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                    <span>New Case</span>
                </a>
            </li>
            <li>
                <a href="<?= BASE_URL ?>pages/reports.php"
                   class="sidebar-link <?= $currentPage === 'reports.php' ? 'active' : '' ?>"
                   id="navReports">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
                    <span>Reports</span>
                </a>
            </li>
            <?php if (isAdmin()): ?>
            <li>
                <a href="<?= BASE_URL ?>pages/issues.php"
                   class="sidebar-link <?= $currentPage === 'issues.php' ? 'active' : '' ?>"
                   id="navIssues">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span>Issues &amp; Logs</span>
                </a>
            </li>
            <?php endif; ?>
        </ul>
    </nav>
    <div class="sidebar-footer">
        <small>v<?= APP_VERSION ?></small>
    </div>
</aside>
