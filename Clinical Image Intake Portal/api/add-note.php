<?php
/**
 * API: Add Support Note (AJAX)
 * Clinical Image Intake Portal
 *
 * Endpoint: POST /api/add-note.php
 * Accepts JSON with case_id, note_body, and note_type.
 * Returns JSON response with the new note data.
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

// Parse input
$input = json_decode(file_get_contents('php://input'), true);
if (!$input) {
    $input = $_POST;
}

$caseId   = (int) ($input['case_id'] ?? 0);
$noteBody = sanitizeInput($input['note_body'] ?? '');
$noteType = sanitizeInput($input['note_type'] ?? 'support');

// Validate
if ($caseId <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Invalid case ID.']);
    exit;
}

if (empty($noteBody)) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Note body is required.']);
    exit;
}

if (!in_array($noteType, getNoteTypes())) {
    $noteType = 'support';
}

// Add the note
try {
    $service = new CaseService();
    $noteId = $service->addSupportNote($caseId, currentUserId(), $noteBody, $noteType);

    echo json_encode([
        'success'     => true,
        'message'     => 'Note added successfully.',
        'note'        => [
            'id'          => $noteId,
            'author_name' => currentUserName(),
            'note_body'   => $noteBody,
            'note_type'   => $noteType,
            'type_label'  => noteTypeLabel($noteType),
            'created_at'  => date('M d, Y g:i A'),
        ],
    ]);
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'message' => 'Server error. Please try again.']);
}
