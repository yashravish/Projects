<?php
/**
 * Mock SOAP Server
 * Clinical Image Intake Portal
 *
 * Provides a SOAP endpoint for clinic/insurance verification.
 * Uses PHP SoapServer in non-WSDL mode.
 *
 * Exposed method:
 *   verifyCoverage(string $clinicName, string $insuranceId) : object
 *
 * To run this SOAP server alongside the main app, start a second
 * PHP built-in server on port 8001:
 *   php -S 127.0.0.1:8001 -t .
 *
 * The main application connects to this endpoint via SoapClient.
 */

// Disable output buffering to avoid interfering with SOAP response
ini_set('display_errors', 0);

/**
 * ClinicVerificationService
 * Provides the verifyCoverage method for the SOAP server.
 */
class ClinicVerificationService
{
    /**
     * Verify clinic coverage and insurance validity.
     *
     * @param string $clinicName   Name of the clinic
     * @param string $insuranceId  Insurance policy ID
     * @return object  Verification result as stdClass
     */
    public function verifyCoverage(string $clinicName, string $insuranceId): object
    {
        // Basic validation
        $approved = !empty($clinicName) && !empty($insuranceId) && strlen($insuranceId) >= 5;

        // Determine policy type based on clinic (for demo variety)
        $policyType = 'Full Coverage';
        if (stripos($clinicName, 'Summit') !== false) {
            $policyType = 'Partial Coverage';
        } elseif (stripos($clinicName, 'Valley') !== false && !$approved) {
            $policyType = 'Limited';
        }

        $result = new stdClass();
        $result->verification_status = $approved ? 'Verified' : 'Not Verified';
        $result->clinic_approved     = $approved ? 1 : 0;
        $result->policy_type         = $approved ? $policyType : 'Unknown';
        $result->message             = $approved
            ? 'Clinic and insurance verified successfully.'
            : 'Unable to verify coverage. Check insurance ID format.';
        $result->checked_at          = date('Y-m-d H:i:s');

        return $result;
    }
}

// ── Handle the SOAP request ─────────────────────────────────
try {
    $server = new SoapServer(null, [
        'uri' => 'urn:ClinicVerificationService',
    ]);
    $server->setClass('ClinicVerificationService');
    $server->handle();
} catch (Exception $e) {
    // If SOAP request fails, return a helpful error
    header('Content-Type: text/plain');
    http_response_code(500);
    echo 'SOAP Server Error: ' . $e->getMessage();
}
