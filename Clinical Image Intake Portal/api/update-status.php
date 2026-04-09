<?php
/**
 * API: Update Case Status (AJAX)
 * Clinical Image Intake Portal
 *
 * Endpoint: POST /api/update-status.php
 * Accepts JSON with case_id, new_status, and optional notes.
 * Returns JSON response.
 */

require_once __DIR__ . '/../includes/auth.php';
require_once __DIR__ . '/../includes/csrf.php';
require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/CaseService.php';

// Require authentication
if (!isLoggedIn()) {
    http_response_code(401);
    header('Content-Type: application/json');
    echo json_encode(['success' => false, 'message' => 'Authentication required.']);
    exit;
}

// Require POST method
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Content-Type: application/json');
    echo json_encode(['success' => false, 'message' => 'Method not allowed.']);
    exit;
}

// Validate CSRF
requireCsrf();

header('Content-Type: application/json');

// Parse JSON body or use POST data
$input = json_decode(file_get_contents('php://input'), true);
if (!$input) {
    $input = $_POST;
}

$caseId    = (int) ($input['case_id'] ?? 0);
$newStatus = sanitizeInput($input['new_status'] ?? '');
$notes     = sanitizeInput($input['notes'] ?? '');

// Validate inputs
if ($caseId <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Invalid case ID.']);
    exit;
}

if (!in_array($newStatus, getStatusOptions())) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Invalid status value.']);
    exit;
}

// Perform the update
try {
    $service = new CaseService();
    $result = $service->updateCaseStatus($caseId, $newStatus, currentUserId(), $notes ?: null);

    if ($result) {
        echo json_encode([
            'success'    => true,
            'message'    => "Status updated to \"{$newStatus}\".",
            'new_status' => $newStatus,
            'badge_class'=> statusBadgeClass($newStatus),
        ]);
    } else {
        http_response_code(400);
        echo json_encode(['success' => false, 'message' => 'Failed to update status.']);
    }
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'message' => 'Server error. Please try again.']);
}
