<?php
/**
 * Logout Handler
 * Clinical Image Intake Portal
 *
 * Destroys the session and redirects to the login page.
 */

require_once __DIR__ . '/includes/auth.php';

// Log the logout
if (isLoggedIn()) {
    require_once __DIR__ . '/services/LogService.php';
    $logger = new LogService();
    $logger->log('info', 'User ' . currentUserName() . ' logged out.', [
        'user_id' => currentUserId(),
    ], 'logout.php', __LINE__);
}

// Destroy session
$_SESSION = [];

if (ini_get('session.use_cookies')) {
    $params = session_get_cookie_params();
    setcookie(
        session_name(),
        '',
        time() - 42000,
        $params['path'],
        $params['domain'],
        $params['secure'],
        $params['httponly']
    );
}

session_destroy();

// Redirect to login
header('Location: ' . BASE_URL . 'login.php');
exit;
