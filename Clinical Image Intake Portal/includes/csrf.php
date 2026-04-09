<?php
/**
 * CSRF Protection Helper
 * Clinical Image Intake Portal
 *
 * Generates and validates CSRF tokens to protect forms and AJAX
 * requests against cross-site request forgery.
 */

/**
 * Generate a new CSRF token and store it in the session.
 * Returns existing token if one already exists for this session.
 *
 * @return string  The CSRF token
 */
function csrfToken(): string
{
    if (empty($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

/**
 * Output a hidden HTML input field containing the CSRF token.
 * Use inside <form> tags.
 */
function csrfField(): void
{
    echo '<input type="hidden" name="csrf_token" value="' . htmlspecialchars(csrfToken(), ENT_QUOTES, 'UTF-8') . '">';
}

/**
 * Validate the CSRF token from the request against the session token.
 * Checks both POST data and the X-CSRF-Token header (for AJAX).
 *
 * @return bool  True if the token is valid
 */
function validateCsrf(): bool
{
    $token = $_POST['csrf_token']
        ?? $_SERVER['HTTP_X_CSRF_TOKEN']
        ?? '';

    if (empty($token) || empty($_SESSION['csrf_token'])) {
        return false;
    }

    return hash_equals($_SESSION['csrf_token'], $token);
}

/**
 * Validate CSRF and abort with 403 if invalid.
 * Use at the top of POST handlers and AJAX endpoints.
 */
function requireCsrf(): void
{
    if (!validateCsrf()) {
        http_response_code(403);
        if (isAjaxRequest()) {
            header('Content-Type: application/json');
            echo json_encode(['success' => false, 'message' => 'Invalid or missing CSRF token.']);
        } else {
            echo '<h2>403 Forbidden</h2><p>Invalid security token. Please go back and try again.</p>';
        }
        exit;
    }
}

/**
 * Check if the current request is an AJAX request.
 */
function isAjaxRequest(): bool
{
    return !empty($_SERVER['HTTP_X_REQUESTED_WITH'])
        && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) === 'xmlhttprequest';
}
