<?php
/**
 * LogService
 * Clinical Image Intake Portal
 *
 * Handles application-level logging to both database (app_logs table)
 * and log files. Used for troubleshooting and admin issue tracking.
 */

require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../config/config.php';

class LogService
{
    private PDO $pdo;

    public function __construct()
    {
        $this->pdo = getDbConnection();
    }

    /**
     * Write a log entry to the database and/or log file.
     *
     * @param string      $level    'info', 'warning', 'error', 'critical'
     * @param string      $message  Log message
     * @param mixed       $context  Additional context data (will be JSON-encoded)
     * @param string|null $file     Source file
     * @param int|null    $line     Source line number
     */
    public function log(string $level, string $message, mixed $context = null, ?string $file = null, ?int $line = null): void
    {
        $contextJson = $context ? (is_string($context) ? $context : json_encode($context)) : null;

        // Write to database
        if (LOG_TO_DB) {
            try {
                $stmt = $this->pdo->prepare(
                    'INSERT INTO app_logs (level, message, context, file, line) VALUES (?, ?, ?, ?, ?)'
                );
                $stmt->execute([$level, $message, $contextJson, $file, $line]);
            } catch (PDOException $e) {
                // Fallback to file-only if DB write fails
                $this->writeToFile('critical', 'Failed to write log to database: ' . $e->getMessage());
            }
        }

        // Write to file
        if (LOG_TO_FILE) {
            $this->writeToFile($level, $message, $contextJson);
        }
    }

    /**
     * Write a log entry to the log file.
     */
    private function writeToFile(string $level, string $message, ?string $context = null): void
    {
        if (!is_dir(LOG_DIR)) {
            @mkdir(LOG_DIR, 0755, true);
        }

        $entry = sprintf(
            "[%s] [%s] %s%s\n",
            date('Y-m-d H:i:s'),
            strtoupper($level),
            $message,
            $context ? ' | Context: ' . $context : ''
        );

        @file_put_contents(LOG_FILE, $entry, FILE_APPEND | LOCK_EX);
    }

    /**
     * Get recent log entries from the database.
     *
     * @param int         $limit  Number of entries to retrieve
     * @param string|null $level  Filter by log level
     * @return array
     */
    public function getRecentLogs(int $limit = 50, ?string $level = null): array
    {
        $sql = 'SELECT * FROM app_logs';
        $params = [];

        if ($level) {
            $sql .= ' WHERE level = ?';
            $params[] = $level;
        }

        $sql .= ' ORDER BY created_at DESC LIMIT ?';
        $params[] = $limit;

        $stmt = $this->pdo->prepare($sql);
        $stmt->execute($params);
        return $stmt->fetchAll();
    }

    /**
     * Get counts of log entries grouped by level.
     */
    public function getLogCountsByLevel(): array
    {
        $stmt = $this->pdo->query(
            'SELECT level, COUNT(*) as count FROM app_logs GROUP BY level ORDER BY FIELD(level, "critical", "error", "warning", "info")'
        );
        return $stmt->fetchAll();
    }

    /**
     * Get recent errors and warnings for the admin issues page.
     */
    public function getRecentIssues(int $limit = 25): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT * FROM app_logs WHERE level IN ("error", "critical", "warning") ORDER BY created_at DESC LIMIT ?'
        );
        $stmt->execute([$limit]);
        return $stmt->fetchAll();
    }
}
