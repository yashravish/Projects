<?php
/**
 * API: SOAP Verification Trigger (AJAX)
 * Clinical Image Intake Portal
 *
 * Endpoint: POST /api/verify-soap.php
 * Triggers SOAP coverage verification for a case.
 * Returns JSON response with verification result.
 */

require_once __DIR__ . '/../includes/auth.php';
require_once __DIR__ . '/../includes/csrf.php';
require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/../services/SoapVerificationService.php';

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

$caseId      = (int) ($input['case_id'] ?? 0);
$clinicName  = sanitizeInput($input['clinic_name'] ?? '');
$insuranceId = sanitizeInput($input['insurance_id'] ?? '');

if ($caseId <= 0) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Invalid case ID.']);
    exit;
}

if (empty($clinicName) || empty($insuranceId)) {
    http_response_code(400);
    echo json_encode(['success' => false, 'message' => 'Clinic name and insurance ID are required for verification.']);
    exit;
}

// Perform SOAP verification
try {
    $service = new SoapVerificationService();
    $result = $service->verifyCoverage($caseId, $clinicName, $insuranceId);

    echo json_encode([
        'success' => true,
        'message' => 'Verification completed.',
        'data'    => $result,
    ]);
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['success' => false, 'message' => 'Verification service error. Please try again.']);
}
