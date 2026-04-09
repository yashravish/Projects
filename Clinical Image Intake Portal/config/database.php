<?php
/**
 * Database Connection (PDO Singleton)
 * Clinical Image Intake Portal
 *
 * Returns a shared PDO instance configured with secure defaults.
 * Uses constants from config.php.
 */

require_once __DIR__ . '/config.php';

/**
 * Get or create the shared PDO database connection.
 *
 * @return PDO
 */
function getDbConnection(): PDO
{
    static $pdo = null;

    if ($pdo === null) {
        $dsn = sprintf(
            'mysql:host=%s;port=%s;dbname=%s;charset=%s',
            DB_HOST,
            DB_PORT,
            DB_NAME,
            DB_CHARSET
        );

        try {
            $pdo = new PDO($dsn, DB_USER, DB_PASS, [
                PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES   => false,
            ]);
        } catch (PDOException $e) {
            // Log to file if possible, show generic error to user
            $errorMsg = date('Y-m-d H:i:s') . ' DB Connection Error: ' . $e->getMessage() . PHP_EOL;
            if (defined('LOG_FILE') && is_writable(dirname(LOG_FILE))) {
                file_put_contents(LOG_FILE, $errorMsg, FILE_APPEND);
            }
            die('<div style="font-family:Arial;padding:40px;color:#c0392b;">'
                . '<h2>Database Connection Error</h2>'
                . '<p>Unable to connect to the database. Please check your configuration and ensure MySQL is running.</p>'
                . '<p>Run <code>setup.php</code> if you have not initialized the database yet.</p>'
                . '</div>');
        }
    }

    return $pdo;
}
