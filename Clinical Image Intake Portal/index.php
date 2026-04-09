<?php
/**
 * Entry Point / Redirect
 * Clinical Image Intake Portal
 *
 * Redirects to the dashboard if logged in, or to the login page.
 */

require_once __DIR__ . '/includes/auth.php';

if (isLoggedIn()) {
    header('Location: ' . BASE_URL . 'pages/dashboard.php');
} else {
    header('Location: ' . BASE_URL . 'login.php');
}
exit;
