<?php
/**
 * Application Configuration
 * Clinical Image Intake Portal
 *
 * Central configuration file for application-wide settings.
 * Adjust BASE_URL if deploying under a subdirectory.
 */

// Prevent direct access
if (!defined('APP_ROOT')) {
    define('APP_ROOT', dirname(__DIR__));
}

// ── Base URL ──────────────────────────────────────────────────
// Set to '/' when serving from document root (PHP built-in server).
// Set to '/clinical-image-intake-portal' when serving from an
// Apache/XAMPP subdirectory under htdocs.
define('BASE_URL', '/');

// ── Application Settings ─────────────────────────────────────
define('APP_NAME', 'Clinical Image Intake Portal');
define('APP_TAGLINE', 'Internal Case Management for Imaging Review Teams');
define('APP_VERSION', '1.0.0');

// ── Database Credentials ─────────────────────────────────────
define('DB_HOST', '127.0.0.1');
define('DB_PORT', '3306');
define('DB_NAME', 'clinical_intake_portal');
define('DB_USER', 'root');
define('DB_PASS', '');          // default XAMPP/MAMP password
define('DB_CHARSET', 'utf8mb4');

// ── Session Settings ─────────────────────────────────────────
define('SESSION_NAME', 'ciip_session');
define('SESSION_LIFETIME', 3600); // 1 hour

// ── Pagination ───────────────────────────────────────────────
define('CASES_PER_PAGE', 10);

// ── SOAP Service ─────────────────────────────────────────────
// URL of the SOAP verification server endpoint.
// If running PHP built-in server on port 8000:
define('SOAP_SERVER_URL', 'http://127.0.0.1:8001/api/soap-server.php');
define('SOAP_SERVICE_URI', 'urn:ClinicVerificationService');

// ── REST External Endpoint ──────────────────────────────────
// URL of the mock external REST endpoint (same project).
define('REST_EXTERNAL_URL', 'http://127.0.0.1:8000/api/rest-external-endpoint.php');

// ── Logging ──────────────────────────────────────────────────
define('LOG_DIR', APP_ROOT . '/logs');
define('LOG_FILE', LOG_DIR . '/app.log');
define('LOG_TO_FILE', true);
define('LOG_TO_DB', true);

// ── Timezone ─────────────────────────────────────────────────
date_default_timezone_set('America/New_York');

// ── Error Display (disable in production) ────────────────────
ini_set('display_errors', 0);
ini_set('log_errors', 1);
error_reporting(E_ALL);
