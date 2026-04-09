<?php
/**
 * API: Sync Case to External System (AJAX)
 * Clinical Image Intake Portal
 *
 * Endpoint: POST /api/sync-case.php
 * Triggers REST sync via IntegrationService.
 * Returns JSON response with sync result.
 */

require_once __DIR__ . '/../includes/auth.php';
require_once __DIR__ . '/../includes/csrf.php';
require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/IntegrationService.php';

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

// Parse input
$input = json_decode(file_get_contents('php://input'), true);
if (!$input) {
    $input = $_POST;
}

$caseId = (int) ($input['case_id'] ?? 0);

if ($caseId <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Invalid case ID.']);
    exit;
}

// Perform sync
try {
    $service = new IntegrationService();
    $result = $service->syncCase($caseId, currentUserId());

    if ($result['success']) {
        echo json_encode([
            'success' => true,
            'message' => $result['message'],
            'data'    => $result['data'] ?? null,
        ]);
    } else {
        http_response_code(200); // Not a server error, just a sync failure
        echo json_encode([
            'success' => false,
            'message' => $result['message'],
        ]);
    }
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'message' => 'Server error during sync. Please try again.']);
}
