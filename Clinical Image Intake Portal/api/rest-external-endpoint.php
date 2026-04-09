<?php
/**
 * Mock REST External Endpoint
 * Clinical Image Intake Portal
 *
 * Simulates an external system that receives case data via REST.
 * This endpoint validates the incoming JSON payload and returns
 * a structured response, mimicking real-world integration behavior.
 *
 * Endpoint: POST /api/rest-external-endpoint.php
 *
 * Expected JSON payload:
 *   {
 *     "case_id": int,
 *     "patient_name": string,
 *     "date_of_birth": string (optional),
 *     "clinic": string,
 *     "provider": string (optional),
 *     "imaging_type": string,
 *     "priority": string,
 *     "status": string,
 *     "insurance_id": string (optional)
 *   }
 *
 * Response (success):
 *   {
 *     "success": true,
 *     "external_reference_id": "EXT-REF-XXXX",
 *     "message": "Case received and registered.",
 *     "received_at": "2026-01-01 12:00:00"
 *   }
 *
 * Response (error):
 *   {
 *     "success": false,
 *     "message": "Error description",
 *     "received_at": "2026-01-01 12:00:00"
 *   }
 */

header('Content-Type: application/json');

// Only accept POST requests
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode([
        'success'     => false,
        'message'     => 'Method not allowed. Use POST.',
        'received_at' => date('Y-m-d H:i:s'),
    ]);
    exit;
}

// Read and decode JSON body
$rawBody = file_get_contents('php://input');
$payload = json_decode($rawBody, true);

if (!$payload) {
    http_response_code(400);
    echo json_encode([
        'success'     => false,
        'message'     => 'Invalid or missing JSON payload.',
        'received_at' => date('Y-m-d H:i:s'),
    ]);
    exit;
}

// Validate required fields
$requiredFields = ['case_id', 'patient_name', 'clinic', 'imaging_type', 'priority', 'status'];
$missingFields = [];

foreach ($requiredFields as $field) {
    if (empty($payload[$field])) {
        $missingFields[] = $field;
    }
}

if (!empty($missingFields)) {
    http_response_code(400);
    echo json_encode([
        'success'     => false,
        'message'     => 'Missing required fields: ' . implode(', ', $missingFields),
        'received_at' => date('Y-m-d H:i:s'),
    ]);
    exit;
}

// Simulate occasional failures for demo realism (case_id divisible by 13)
if ($payload['case_id'] % 13 === 0) {
    http_response_code(500);
    echo json_encode([
        'success'     => false,
        'message'     => 'External system temporarily unavailable. Please retry.',
        'received_at' => date('Y-m-d H:i:s'),
    ]);
    exit;
}

// Generate a mock external reference ID
$externalRefId = 'EXT-REF-' . str_pad((string) $payload['case_id'], 4, '0', STR_PAD_LEFT);

// Success response
echo json_encode([
    'success'               => true,
    'external_reference_id' => $externalRefId,
    'message'               => 'Case received and registered.',
    'received_at'           => date('Y-m-d H:i:s'),
]);
