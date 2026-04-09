<?php
/**
 * Helper Functions
 * Clinical Image Intake Portal
 *
 * Shared utility functions used across the application:
 * sanitization, validation, formatting, flash messages, etc.
 */

// ── Output Escaping ──────────────────────────────────────────

/**
 * Escape a string for safe HTML output (XSS prevention).
 */
function h(mixed $value): string
{
    if ($value === null) return '';
    return htmlspecialchars((string) $value, ENT_QUOTES, 'UTF-8');
}

// ── Flash Messages ───────────────────────────────────────────

/**
 * Set a flash message in the session.
 *
 * @param string $type     'success', 'danger', 'warning', 'info'
 * @param string $message  Message text
 */
function setFlash(string $type, string $message): void
{
    $_SESSION['flash'] = [
        'type'    => $type,
        'message' => $message,
    ];
}

/**
 * Retrieve and clear the flash message.
 *
 * @return array|null  ['type' => ..., 'message' => ...] or null
 */
function getFlash(): ?array
{
    if (isset($_SESSION['flash'])) {
        $flash = $_SESSION['flash'];
        unset($_SESSION['flash']);
        return $flash;
    }
    return null;
}

// ── Redirect ─────────────────────────────────────────────────

/**
 * Redirect to a URL relative to BASE_URL.
 */
function redirect(string $path): void
{
    header('Location: ' . BASE_URL . ltrim($path, '/'));
    exit;
}

// ── Validation Helpers ───────────────────────────────────────

/**
 * Check if a string is non-empty after trimming.
 */
function isRequired(mixed $value): bool
{
    return is_string($value) && trim($value) !== '';
}

/**
 * Validate an email address.
 */
function isValidEmail(string $email): bool
{
    return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;
}

/**
 * Validate a date string (YYYY-MM-DD).
 */
function isValidDate(string $date): bool
{
    $d = DateTime::createFromFormat('Y-m-d', $date);
    return $d && $d->format('Y-m-d') === $date;
}

/**
 * Validate a phone number (basic: digits, dashes, spaces, parens).
 */
function isValidPhone(string $phone): bool
{
    return (bool) preg_match('/^[\d\s\-\(\)\+\.]{7,20}$/', $phone);
}

// ── Sanitization ─────────────────────────────────────────────

/**
 * Trim and sanitize a string input.
 */
function sanitizeInput(mixed $value): string
{
    if ($value === null) return '';
    return trim(strip_tags((string) $value));
}

// ── Formatting ───────────────────────────────────────────────

/**
 * Format a date for display (M d, Y).
 */
function formatDate(?string $date): string
{
    if (empty($date)) return '—';
    $dt = new DateTime($date);
    return $dt->format('M d, Y');
}

/**
 * Format a datetime for display (M d, Y g:i A).
 */
function formatDateTime(?string $datetime): string
{
    if (empty($datetime)) return '—';
    $dt = new DateTime($datetime);
    return $dt->format('M d, Y g:i A');
}

/**
 * Calculate age from date of birth.
 */
function calculateAge(string $dob): int
{
    $birth = new DateTime($dob);
    $now   = new DateTime();
    return (int) $birth->diff($now)->y;
}

// ── Status & Priority Badges ─────────────────────────────────

/**
 * Return CSS class for a status badge.
 */
function statusBadgeClass(string $status): string
{
    return match ($status) {
        'New'                       => 'badge-new',
        'Under Review'              => 'badge-review',
        'Awaiting Clinic Response'  => 'badge-awaiting',
        'Verified'                  => 'badge-verified',
        'Escalated'                 => 'badge-escalated',
        'Closed'                    => 'badge-closed',
        default                     => 'badge-default',
    };
}

/**
 * Return CSS class for a priority badge.
 */
function priorityBadgeClass(string $priority): string
{
    return match ($priority) {
        'Low'    => 'badge-low',
        'Medium' => 'badge-medium',
        'High'   => 'badge-high',
        'Urgent' => 'badge-urgent',
        default  => 'badge-default',
    };
}

/**
 * Return CSS class for a sync status badge.
 */
function syncBadgeClass(string $status): string
{
    return match ($status) {
        'synced'     => 'badge-verified',
        'failed'     => 'badge-escalated',
        'not_synced' => 'badge-default',
        default      => 'badge-default',
    };
}

// ── Allowed Values ───────────────────────────────────────────

/**
 * Get valid status options.
 */
function getStatusOptions(): array
{
    return ['New', 'Under Review', 'Awaiting Clinic Response', 'Verified', 'Escalated', 'Closed'];
}

/**
 * Get valid priority options.
 */
function getPriorityOptions(): array
{
    return ['Low', 'Medium', 'High', 'Urgent'];
}

/**
 * Get valid imaging type options.
 */
function getImagingTypes(): array
{
    return ['Skin Lesion', 'Facial Analysis', 'Scar Review', 'Follow-Up', 'Other'];
}

/**
 * Get valid support note type options.
 */
function getNoteTypes(): array
{
    return ['support', 'technical', 'customer_issue', 'sync_issue'];
}

/**
 * Get a human-readable label for a note type.
 */
function noteTypeLabel(string $type): string
{
    return match ($type) {
        'support'        => 'Support',
        'technical'      => 'Technical',
        'customer_issue' => 'Customer Issue',
        'sync_issue'     => 'Sync Issue',
        default          => ucfirst($type),
    };
}

// ── User List ────────────────────────────────────────────────

/**
 * Get all users as id => full_name for assignment dropdowns.
 */
function getUserList(): array
{
    $pdo = getDbConnection();
    $stmt = $pdo->query('SELECT id, full_name FROM users ORDER BY full_name');
    $users = [];
    while ($row = $stmt->fetch()) {
        $users[$row['id']] = $row['full_name'];
    }
    return $users;
}
