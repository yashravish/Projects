@echo off
echo ========================================
echo  Signal Desk Lite - Railway Deployment
echo ========================================
echo.

REM Navigate to API directory
cd /d "%~dp0"

echo Step 1: Verifying Railway login...
railway whoami
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Not logged in to Railway!
    echo Please run: railway login
    exit /b 1
)
echo.

echo Step 2: Initializing Railway project...
railway init
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to initialize Railway project
    exit /b 1
)
echo.

echo Step 3: Linking to Railway environment...
railway environment
echo.

echo Step 4: Setting up environment variables...
echo You need to set these variables in Railway:
echo - DATABASE_URL (Neon PostgreSQL connection string)
echo - DIRECT_URL (Neon direct connection for migrations)
echo - REDIS_URL (Upstash Redis connection string)
echo - UPSTASH_REDIS_REST_URL
echo - UPSTASH_REDIS_REST_TOKEN
echo - JWT_SECRET
echo - OPENAI_API_KEY (optional)
echo - PORT=3001
echo - API_HOST=0.0.0.0
echo - NODE_ENV=production
echo.
echo Open Railway dashboard to set variables:
railway open
echo.
pause

echo Step 5: Deploying to Railway...
railway up
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Deployment failed
    exit /b 1
)
echo.

echo ========================================
echo  Deployment Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Get your Railway service URL: railway domain
echo 2. Update Vercel with NEXT_PUBLIC_API_URL
echo.
railway status
echo.
pause
