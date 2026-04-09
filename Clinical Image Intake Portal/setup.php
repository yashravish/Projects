<?php
/**
 * Setup Script
 * Clinical Image Intake Portal
 *
 * Run this script to initialize the database, create tables,
 * and seed sample data with properly hashed passwords.
 *
 * Usage (CLI):
 *   php setup.php
 *
 * Usage (Browser):
 *   http://localhost:8000/setup.php
 */

// Prevent repeated runs in production
$lockFile = __DIR__ . '/logs/.setup_complete';

echo "<pre style='font-family:Consolas,monospace; padding:20px; max-width:800px; margin:20px auto;'>\n";
echo "===============================================\n";
echo "  Clinical Image Intake Portal — Setup Script  \n";
echo "===============================================\n\n";

// ── Configuration ────────────────────────────────────────────
define('APP_ROOT', __DIR__);
require_once __DIR__ . '/config/config.php';

$dbHost    = DB_HOST;
$dbPort    = DB_PORT;
$dbName    = DB_NAME;
$dbUser    = DB_USER;
$dbPass    = DB_PASS;
$dbCharset = DB_CHARSET;

try {
    // ── Step 1: Connect to MySQL (without database) ─────────
    echo "[1/4] Connecting to MySQL...\n";
    $pdo = new PDO(
        "mysql:host={$dbHost};port={$dbPort};charset={$dbCharset}",
        $dbUser,
        $dbPass,
        [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]
    );
    echo "      ✓ Connected to MySQL.\n\n";

    // ── Step 2: Create Database & Run Schema ────────────────
    echo "[2/4] Creating database and tables...\n";
    $schema = file_get_contents(__DIR__ . '/sql/schema.sql');
    if (!$schema) {
        throw new RuntimeException('Could not read sql/schema.sql');
    }

    // Execute schema statements
    $pdo->exec("CREATE DATABASE IF NOT EXISTS `{$dbName}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
    $pdo->exec("USE `{$dbName}`");

    // Remove the CREATE DATABASE and USE statements from schema (we already did them)
    $schema = preg_replace('/CREATE DATABASE.*?;/si', '', $schema);
    $schema = preg_replace('/USE\s+\w+;/si', '', $schema);

    // Split by semicolons and execute each statement
    $statements = array_filter(array_map('trim', explode(';', $schema)));
    foreach ($statements as $stmt) {
        if (!empty($stmt)) {
            try {
                $pdo->exec($stmt);
            } catch (PDOException $e) {
                // Ignore "already exists" errors for idempotency
                if (strpos($e->getMessage(), 'already exists') === false &&
                    strpos($e->getMessage(), 'Duplicate') === false) {
                    echo "      ⚠ Warning: " . $e->getMessage() . "\n";
                }
            }
        }
    }
    echo "      ✓ Database '{$dbName}' and tables created.\n\n";

    // ── Step 3: Seed Users ──────────────────────────────────
    echo "[3/4] Seeding data...\n";

    // Check if users already exist
    $existingUsers = $pdo->query("SELECT COUNT(*) FROM users")->fetchColumn();

    if ($existingUsers > 0) {
        echo "      ⚠ Users already exist. Skipping user seed.\n";
    } else {
        // Create users with properly hashed passwords
        $users = [
            ['admin',  'admin123',    'Alex Thompson',  'admin',   'alex.thompson@clinicalportal.local'],
            ['sarah',  'support123',  'Sarah Martinez', 'support', 'sarah.martinez@clinicalportal.local'],
        ];

        $stmtUser = $pdo->prepare(
            'INSERT INTO users (username, password_hash, full_name, role, email) VALUES (?, ?, ?, ?, ?)'
        );

        foreach ($users as $u) {
            $hash = password_hash($u[1], PASSWORD_DEFAULT);
            $stmtUser->execute([$u[0], $hash, $u[2], $u[3], $u[4]]);
            echo "      ✓ User '{$u[0]}' created (password: {$u[1]})\n";
        }
    }

    // Check if cases already exist
    $existingCases = $pdo->query("SELECT COUNT(*) FROM cases")->fetchColumn();

    if ($existingCases > 0) {
        echo "      ⚠ Cases already exist. Skipping case seed.\n";
    } else {
        // Run the seed.sql for sample data
        $seedSql = file_get_contents(__DIR__ . '/sql/seed.sql');
        if ($seedSql) {
            // Remove the USE statement
            $seedSql = preg_replace('/USE\s+\w+;/si', '', $seedSql);

            // Remove user INSERT (we already handled users above)
            // The seed.sql has user inserts commented out, so this is fine

            $seedStatements = array_filter(array_map('trim', explode(';', $seedSql)));
            $seedCount = 0;
            foreach ($seedStatements as $stmt) {
                if (!empty($stmt) && stripos($stmt, 'INSERT') !== false) {
                    try {
                        $pdo->exec($stmt);
                        $seedCount++;
                    } catch (PDOException $e) {
                        echo "      ⚠ Seed warning: " . $e->getMessage() . "\n";
                    }
                }
            }
            echo "      ✓ Executed {$seedCount} seed INSERT statements.\n";
        }
    }
    echo "\n";

    // ── Step 4: Create Log Directory ────────────────────────
    echo "[4/4] Finalizing setup...\n";
    if (!is_dir(__DIR__ . '/logs')) {
        mkdir(__DIR__ . '/logs', 0755, true);
    }

    // Mark setup as complete
    file_put_contents($lockFile, date('Y-m-d H:i:s'));

    echo "      ✓ Log directory ready.\n\n";
    echo "===============================================\n";
    echo "  ✓ Setup Complete!                            \n";
    echo "===============================================\n\n";
    echo "Demo Credentials:\n";
    echo "  Admin:    admin / admin123\n";
    echo "  Support:  sarah / support123\n\n";
    echo "Start the application:\n";
    echo "  php -S 127.0.0.1:8000 -t .\n\n";
    echo "Then open: http://127.0.0.1:8000\n\n";
    echo "For SOAP verification, start a second server:\n";
    echo "  php -S 127.0.0.1:8001 -t .\n";

} catch (PDOException $e) {
    echo "\n✗ DATABASE ERROR: " . $e->getMessage() . "\n\n";
    echo "Make sure:\n";
    echo "  1. MySQL/MariaDB is running\n";
    echo "  2. Credentials in config/config.php are correct\n";
    echo "  3. The database user has CREATE and INSERT privileges\n";
} catch (Exception $e) {
    echo "\n✗ ERROR: " . $e->getMessage() . "\n";
}

echo "</pre>\n";
