<?php
/**
 * Authentication Helper
 * Clinical Image Intake Portal
 *
 * Handles session initialization, login verification, and access control.
 * Include this file at the top of every protected page.
 */

require_once __DIR__ . '/../config/config.php';

// ── Session Hardening ────────────────────────────────────────
if (session_status() === PHP_SESSION_NONE) {
    session_name(SESSION_NAME);
    session_set_cookie_params([
        'lifetime' => SESSION_LIFETIME,
        'path'     => '/',
        'secure'   => false,   // set true in production with HTTPS
        'httponly'  => true,
        'samesite'  => 'Lax',
    ]);
    session_start();
}

/**
 * Check if the current user is authenticated.
 */
function isLoggedIn(): bool
{
    return isset($_SESSION['logged_in']) && $_SESSION['logged_in'] === true;
}

/**
 * Require authentication. Redirects to login if not authenticated.
 */
function requireLogin(): void
{
    if (!isLoggedIn()) {
        $_SESSION['flash'] = [
            'type'    => 'warning',
            'message' => 'Please log in to access that page.',
        ];
        header('Location: ' . BASE_URL . 'login.php');
        exit;
    }
}

/**
 * Require a specific role. Redirects with error if unauthorized.
 *
 * @param string $role  Required role ('admin' or 'support')
 */
function requireRole(string $role): void
{
    requireLogin();
    if ($_SESSION['role'] !== $role) {
        $_SESSION['flash'] = [
            'type'    => 'danger',
            'message' => 'You do not have permission to access that page.',
        ];
        header('Location: ' . BASE_URL . 'pages/dashboard.php');
        exit;
    }
}

/**
 * Get the currently logged-in user's ID.
 */
function currentUserId(): ?int
{
    return $_SESSION['user_id'] ?? null;
}

/**
 * Get the currently logged-in user's full name.
 */
function currentUserName(): string
{
    return $_SESSION['full_name'] ?? 'Unknown';
}

/**
 * Get the currently logged-in user's role.
 */
function currentUserRole(): string
{
    return $_SESSION['role'] ?? '';
}

/**
 * Check if current user is an admin.
 */
function isAdmin(): bool
{
    return currentUserRole() === 'admin';
}
