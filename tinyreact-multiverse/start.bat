@echo off
REM TinyReact Multiverse Startup Script for Windows

echo.
echo ================================================
echo   TinyReact Multiverse
echo ================================================
echo.

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python found
    echo [*] Starting server on http://localhost:8000
    echo.
    echo Press Ctrl+C to stop the server
    echo In your browser, press Alt+D to open the Multiverse Dev Pane
    echo.
    python -m http.server 8000
    goto :end
)

REM Check for Node.js
where node >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Node.js found
    echo [*] Starting server on http://localhost:3000
    echo.
    echo Press Ctrl+C to stop the server
    echo In your browser, press Alt+D to open the Multiverse Dev Pane
    echo.
    npx serve -l 3000
    goto :end
)

echo [ERROR] No suitable HTTP server found
echo.
echo Please install one of the following:
echo   - Python 3: https://www.python.org/
echo   - Node.js: https://nodejs.org/
echo.
echo Or use any other static file server
pause
exit /b 1

:end
