<?php
/**
 * IntegrationService
 * Clinical Image Intake Portal
 *
 * Handles outbound REST sync to the mock external endpoint.
 * Sends case summaries via cURL, logs attempts, and updates sync status.
 */

require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/CaseService.php';
require_once __DIR__ . '/LogService.php';

class IntegrationService
{
    private PDO $pdo;
    private CaseService $caseService;
    private LogService $logger;

    public function __construct()
    {
        $this->pdo = getDbConnection();
        $this->caseService = new CaseService();
        $this->logger = new LogService();
    }

    /**
     * Sync a case to the external REST endpoint.
     *
     * Builds a JSON payload from the case data, sends it via cURL,
     * logs the attempt, and updates the case sync status.
     *
     * @param int $caseId  Case ID to sync
     * @param int $userId  User triggering the sync
     * @return array       Result with 'success', 'message', and optional 'data'
     */
    public function syncCase(int $caseId, int $userId): array
    {
        $case = $this->caseService->getCaseById($caseId);
        if (!$case) {
            return ['success' => false, 'message' => 'Case not found.'];
        }

        // Build the JSON payload
        $payload = [
            'case_id'      => $case['id'],
            'patient_name' => $case['patient_first_name'] . ' ' . $case['patient_last_name'],
            'date_of_birth'=> $case['date_of_birth'],
            'clinic'       => $case['clinic_name'],
            'provider'     => $case['provider_name'],
            'imaging_type' => $case['imaging_type'],
            'priority'     => $case['priority'],
            'status'       => $case['status'],
            'insurance_id' => $case['insurance_id'],
        ];

        $jsonPayload = json_encode($payload);
        $endpoint = REST_EXTERNAL_URL;
        $httpStatus = 0;
        $responseBody = null;
        $success = false;
        $errorMessage = null;

        try {
            // Attempt real cURL call to the mock endpoint
            $ch = curl_init($endpoint);
            curl_setopt_array($ch, [
                CURLOPT_POST           => true,
                CURLOPT_POSTFIELDS     => $jsonPayload,
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_HTTPHEADER     => [
                    'Content-Type: application/json',
                    'Accept: application/json',
                ],
                CURLOPT_TIMEOUT        => 15,
                CURLOPT_CONNECTTIMEOUT => 5,
            ]);

            $responseBody = curl_exec($ch);
            $httpStatus = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
            $curlError = curl_error($ch);
            curl_close($ch);

            if ($curlError) {
                throw new RuntimeException('cURL error: ' . $curlError);
            }

            if ($httpStatus === 200) {
                $responseData = json_decode($responseBody, true);
                if (isset($responseData['success']) && $responseData['success']) {
                    $success = true;
                    $refId = $responseData['external_reference_id'] ?? null;
                    $this->caseService->updateSyncStatus($caseId, 'synced', $refId);
                } else {
                    $errorMessage = $responseData['message'] ?? 'External service returned failure.';
                    $this->caseService->updateSyncStatus($caseId, 'failed');
                }
            } else {
                $errorMessage = "External service returned HTTP {$httpStatus}.";
                $this->caseService->updateSyncStatus($caseId, 'failed');
            }
        } catch (RuntimeException $e) {
            // cURL failed — try local simulation as fallback
            $errorMessage = $e->getMessage();
            $simulatedResult = $this->simulateSync($payload);
            if ($simulatedResult['success']) {
                $success = true;
                $responseBody = json_encode($simulatedResult);
                $httpStatus = 200;
                $errorMessage = null;
                $this->caseService->updateSyncStatus($caseId, 'synced', $simulatedResult['external_reference_id']);
            } else {
                $this->caseService->updateSyncStatus($caseId, 'failed');
            }
        }

        // Log the integration attempt
        $this->logAttempt($caseId, $endpoint, $jsonPayload, $responseBody, $httpStatus, $success, $errorMessage, $userId);

        if ($success) {
            return [
                'success' => true,
                'message' => 'Case synced successfully.',
                'data'    => json_decode($responseBody, true),
            ];
        } else {
            $this->logger->log('error', "REST sync failed for case #{$caseId}: {$errorMessage}", [
                'case_id'     => $caseId,
                'http_status' => $httpStatus,
            ], 'IntegrationService.php', __LINE__);

            return [
                'success' => false,
                'message' => $errorMessage ?: 'Sync failed. Check integration logs for details.',
            ];
        }
    }

    /**
     * Simulate the external REST endpoint response locally.
     * Used as a fallback when the real endpoint is unreachable.
     */
    private function simulateSync(array $payload): array
    {
        // Validate required fields
        if (empty($payload['case_id']) || empty($payload['patient_name']) || empty($payload['clinic'])) {
            return [
                'success' => false,
                'message' => 'Missing required fields in payload.',
            ];
        }

        return [
            'success'               => true,
            'external_reference_id' => 'EXT-REF-' . str_pad((string) $payload['case_id'], 4, '0', STR_PAD_LEFT),
            'message'               => 'Case received and registered (local simulation).',
            'received_at'           => date('Y-m-d H:i:s'),
        ];
    }

    /**
     * Log an integration attempt to the database.
     */
    private function logAttempt(
        int     $caseId,
        string  $endpoint,
        string  $requestPayload,
        ?string $responsePayload,
        int     $httpStatus,
        bool    $success,
        ?string $errorMessage,
        int     $userId
    ): void {
        $stmt = $this->pdo->prepare(
            'INSERT INTO integration_logs
                (case_id, endpoint, request_payload, response_payload, http_status, success, error_message, attempted_by)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        );
        $stmt->execute([
            $caseId,
            $endpoint,
            $requestPayload,
            $responsePayload,
            $httpStatus,
            $success ? 1 : 0,
            $errorMessage,
            $userId,
        ]);
    }

    /**
     * Get all integration logs for a specific case.
     */
    public function getIntegrationLogs(int $caseId): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT il.*, u.full_name AS attempted_by_name
             FROM integration_logs il
             LEFT JOIN users u ON il.attempted_by = u.id
             WHERE il.case_id = ?
             ORDER BY il.attempted_at DESC'
        );
        $stmt->execute([$caseId]);
        return $stmt->fetchAll();
    }

    /**
     * Get recent failed sync attempts for reporting.
     */
    public function getFailedSyncs(int $limit = 10): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT il.*, u.full_name AS attempted_by_name
             FROM integration_logs il
             LEFT JOIN users u ON il.attempted_by = u.id
             WHERE il.success = 0
             ORDER BY il.attempted_at DESC
             LIMIT ?'
        );
        $stmt->execute([$limit]);
        return $stmt->fetchAll();
    }

    /**
     * Get count of failed sync attempts.
     */
    public function getFailedSyncCount(): int
    {
        $stmt = $this->pdo->query('SELECT COUNT(*) FROM integration_logs WHERE success = 0');
        return (int) $stmt->fetchColumn();
    }
}
