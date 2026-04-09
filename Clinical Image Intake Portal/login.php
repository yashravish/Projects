<?php
/**
 * Login Page
 * Clinical Image Intake Portal
 *
 * Handles authentication with session-based login.
 * Validates credentials against the users table with password_verify().
 */

require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/functions.php';
require_once __DIR__ . '/config/database.php';

// Redirect if already logged in
if (isLoggedIn()) {
    header('Location: ' . BASE_URL . 'pages/dashboard.php');
    exit;
}

$error = '';

// ── Handle Login Submission ─────────────────────────────────
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!validateCsrf()) {
        $error = 'Invalid security token. Please try again.';
    } else {
        $username = sanitizeInput($_POST['username'] ?? '');
        $password = $_POST['password'] ?? '';

        if (empty($username) || empty($password)) {
            $error = 'Username and password are required.';
        } else {
            try {
                $pdo = getDbConnection();
                $stmt = $pdo->prepare('SELECT * FROM users WHERE username = ?');
                $stmt->execute([$username]);
                $user = $stmt->fetch();

                if ($user && password_verify($password, $user['password_hash'])) {
                    // Regenerate session ID to prevent fixation
                    session_regenerate_id(true);

                    $_SESSION['logged_in'] = true;
                    $_SESSION['user_id']   = (int) $user['id'];
                    $_SESSION['username']  = $user['username'];
                    $_SESSION['full_name'] = $user['full_name'];
                    $_SESSION['role']      = $user['role'];

                    // Log the login
                    require_once __DIR__ . '/services/LogService.php';
                    $logger = new LogService();
                    $logger->log('info', "User {$user['username']} logged in.", [
                        'user_id' => $user['id'],
                    ], 'login.php', __LINE__);

                    header('Location: ' . BASE_URL . 'pages/dashboard.php');
                    exit;
                } else {
                    $error = 'Invalid username or password.';
                }
            } catch (PDOException $e) {
                $error = 'A system error occurred. Please try again later.';
            }
        }
    }
}

$flash = getFlash();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Login - Clinical Image Intake Portal">
    <title>Login — <?= APP_NAME ?></title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="<?= BASE_URL ?>assets/css/styles.css">
</head>
<body class="login-page">
    <div class="login-container">
        <div class="login-card" id="loginCard">
            <div class="login-header">
                <div class="login-icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <h1><?= h(APP_NAME) ?></h1>
                <p class="login-tagline"><?= h(APP_TAGLINE) ?></p>
            </div>

            <?php if ($flash): ?>
            <div class="alert alert-<?= h($flash['type']) ?>">
                <?= h($flash['message']) ?>
            </div>
            <?php endif; ?>

            <?php if ($error): ?>
            <div class="alert alert-danger" id="loginError">
                <?= h($error) ?>
            </div>
            <?php endif; ?>

            <form method="POST" action="" id="loginForm">
                <input type="hidden" name="csrf_token" value="<?= csrfToken() ?>">

                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" class="form-control"
                           placeholder="Enter your username"
                           value="<?= h($username ?? '') ?>"
                           required autofocus autocomplete="username">
                </div>

                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" class="form-control"
                           placeholder="Enter your password"
                           required autocomplete="current-password">
                </div>

                <button type="submit" class="btn btn-primary btn-block" id="btnLogin">Sign In</button>
            </form>

            <div class="login-footer">
                <small>Authorized personnel only. All access is monitored and logged.</small>
            </div>
        </div>
    </div>
</body>
</html>
