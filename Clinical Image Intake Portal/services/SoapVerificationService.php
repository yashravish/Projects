<?php
/**
 * SoapVerificationService
 * Clinical Image Intake Portal
 *
 * Calls the mock SOAP server to verify clinic/insurance coverage.
 * Uses PHP's SoapClient in non-WSDL mode.
 * Falls back to local simulation if the SOAP server is unreachable.
 */

require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/CaseService.php';
require_once __DIR__ . '/LogService.php';

class SoapVerificationService
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
     * Verify clinic coverage via SOAP for a given case.
     *
     * @param int    $caseId      Case ID
     * @param string $clinicName  Clinic name to verify
     * @param string $insuranceId Insurance ID to verify
     * @return array  Verification result
     */
    public function verifyCoverage(int $caseId, string $clinicName, string $insuranceId): array
    {
        $result = null;

        // Attempt real SOAP call
        if (extension_loaded('soap')) {
            try {
                $client = new SoapClient(null, [
                    'location'           => SOAP_SERVER_URL,
                    'uri'                => SOAP_SERVICE_URI,
                    'trace'              => true,
                    'connection_timeout' => 5,
                    'exceptions'         => true,
                ]);

                $response = $client->verifyCoverage($clinicName, $insuranceId);

                // SoapClient returns stdClass; convert to array
                $result = (array) $response;

            } catch (SoapFault $e) {
                $this->logger->log('warning', 'SOAP call failed: ' . $e->getMessage(), [
                    'case_id'      => $caseId,
                    'clinic'       => $clinicName,
                    'insurance_id' => $insuranceId,
                ], 'SoapVerificationService.php', __LINE__);

                // Fall back to local simulation
                $result = $this->simulateVerification($clinicName, $insuranceId);
                $result['_note'] = 'SOAP server unreachable; result simulated locally.';

            } catch (\Exception $e) {
                $this->logger->log('error', 'SOAP client error: ' . $e->getMessage(), [
                    'case_id' => $caseId,
                ], 'SoapVerificationService.php', __LINE__);

                $result = $this->simulateVerification($clinicName, $insuranceId);
                $result['_note'] = 'SOAP server unreachable; result simulated locally.';
            }
        } else {
            // SOAP extension not loaded — simulate
            $this->logger->log('warning', 'PHP SOAP extension not loaded. Using local simulation.', [
                'case_id' => $caseId,
            ], 'SoapVerificationService.php', __LINE__);

            $result = $this->simulateVerification($clinicName, $insuranceId);
            $result['_note'] = 'php-soap extension not enabled; result simulated locally.';
        }

        // Store result in database
        $this->storeVerification($caseId, $clinicName, $insuranceId, $result);

        // Update case SOAP status
        $this->caseService->updateSoapStatus($caseId, $result['verification_status'] ?? 'Unknown');

        return $result;
    }

    /**
     * Simulate the SOAP verification response locally.
     * Mimics the logic in the real SOAP server.
     */
    private function simulateVerification(string $clinicName, string $insuranceId): array
    {
        $approved = !empty($clinicName) && !empty($insuranceId) && strlen($insuranceId) >= 5;

        // Some clinics get partial coverage for variety
        $policyType = 'Full Coverage';
        if (stripos($clinicName, 'Summit') !== false) {
            $policyType = 'Partial Coverage';
        }

        return [
            'verification_status' => $approved ? 'Verified' : 'Not Verified',
            'clinic_approved'     => $approved ? 1 : 0,
            'policy_type'         => $approved ? $policyType : 'Unknown',
            'message'             => $approved
                ? 'Clinic and insurance verified successfully.'
                : 'Unable to verify coverage. Check insurance ID.',
            'checked_at'          => date('Y-m-d H:i:s'),
        ];
    }

    /**
     * Store a verification result in the soap_verifications table.
     */
    private function storeVerification(int $caseId, string $clinicName, string $insuranceId, array $result): void
    {
        $stmt = $this->pdo->prepare(
            'INSERT INTO soap_verifications
                (case_id, clinic_name, insurance_id, verification_status, clinic_approved, policy_type, message, checked_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        );
        $stmt->execute([
            $caseId,
            $clinicName,
            $insuranceId,
            $result['verification_status'] ?? 'Unknown',
            $result['clinic_approved'] ?? 0,
            $result['policy_type'] ?? null,
            $result['message'] ?? null,
            $result['checked_at'] ?? date('Y-m-d H:i:s'),
        ]);
    }

    /**
     * Get the most recent SOAP verification for a case.
     */
    public function getVerification(int $caseId): ?array
    {
        $stmt = $this->pdo->prepare(
            'SELECT * FROM soap_verifications WHERE case_id = ? ORDER BY created_at DESC LIMIT 1'
        );
        $stmt->execute([$caseId]);
        $row = $stmt->fetch();
        return $row ?: null;
    }

    /**
     * Get all SOAP verifications for a case (history).
     */
    public function getVerificationHistory(int $caseId): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT * FROM soap_verifications WHERE case_id = ? ORDER BY created_at DESC'
        );
        $stmt->execute([$caseId]);
        return $stmt->fetchAll();
    }
}
