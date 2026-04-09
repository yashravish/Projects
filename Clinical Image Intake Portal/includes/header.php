<?php
/**
 * Header Partial
 * Clinical Image Intake Portal
 *
 * Shared HTML head & top navigation bar.
 * Expects $pageTitle to be set before including this file.
 */

require_once __DIR__ . '/auth.php';
require_once __DIR__ . '/csrf.php';
require_once __DIR__ . '/functions.php';
require_once __DIR__ . '/../config/database.php';

$pageTitle = $pageTitle ?? 'Dashboard';
$flash = getFlash();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Clinical Image Intake Portal - Internal Case Management for Imaging Review Teams">
    <title><?= h($pageTitle) ?> — <?= h(APP_NAME) ?></title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="<?= BASE_URL ?>assets/css/styles.css">
    <script>
        // Make CSRF token and base URL available to JavaScript
        var APP = {
            baseUrl: '<?= rtrim(BASE_URL, '/') ?>',
            csrfToken: '<?= csrfToken() ?>'
        };
    </script>
</head>
<body>
    <!-- Top Navigation Bar -->
    <header class="top-nav" id="topNav">
        <div class="top-nav-brand">
            <span class="brand-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </span>
            <div class="brand-text">
                <span class="brand-name"><?= h(APP_NAME) ?></span>
                <span class="brand-tagline"><?= h(APP_TAGLINE) ?></span>
            </div>
        </div>
        <?php if (isLoggedIn()): ?>
        <div class="top-nav-user">
            <span class="user-role-badge"><?= h(ucfirst(currentUserRole())) ?></span>
            <span class="user-name"><?= h(currentUserName()) ?></span>
            <a href="<?= BASE_URL ?>logout.php" class="btn btn-sm btn-outline" id="logoutBtn">Logout</a>
        </div>
        <?php endif; ?>
    </header>

    <div class="app-container">
        <?php if (isLoggedIn()): ?>
            <?php include __DIR__ . '/sidebar.php'; ?>
        <?php endif; ?>

        <main class="main-content" id="mainContent">
            <?php if ($flash): ?>
            <div class="alert alert-<?= h($flash['type']) ?>" id="flashAlert">
                <?= h($flash['message']) ?>
                <button type="button" class="alert-close" onclick="this.parentElement.remove()">×</button>
            </div>
            <?php endif; ?>
