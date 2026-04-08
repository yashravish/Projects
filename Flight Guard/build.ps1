# PowerShell Build Script for FlightGuard
# Use this on Windows when GNU Make is not available.
#
# Usage:
#   .\build.ps1              # Build demo
#   .\build.ps1 test         # Build and run tests
#   .\build.ps1 sim          # Build and run simulator
#   .\build.ps1 run          # Build and run demo
#   .\build.ps1 clean        # Remove build artifacts
#   .\build.ps1 all          # Build all executables
#   .\build.ps1 coverage     # Build with coverage and run tests

param(
    [string]$Target = "build"
)

$ErrorActionPreference = "Stop"

# Configuration
$CC       = "gcc"
$CFLAGS   = "-std=c99 -Wall -Wextra -Werror -pedantic -Iinclude"
$BuildDir = "build"

# Source files
$CoreSrc  = "src/flightguard.c"
$MainSrc  = "src/main.c"
$SimSrc   = "sim/simulator.c"
$TestSrcs = "test/test_main.c test/test_unit.c test/test_integration.c test/test_fault_injection.c"

# Executables
$Demo      = "$BuildDir/flightguard_demo.exe"
$Tests     = "$BuildDir/flightguard_tests.exe"
$Simulator = "$BuildDir/flightguard_sim.exe"

function Ensure-BuildDir {
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }
}

function Build-Demo {
    Ensure-BuildDir
    Write-Host "Building demo..." -ForegroundColor Cyan
    $cmd = "$CC $CFLAGS -o $Demo $CoreSrc $MainSrc"
    Write-Host "  $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    Write-Host "  Built: $Demo" -ForegroundColor Green
}

function Build-Tests {
    Ensure-BuildDir
    Write-Host "Building tests..." -ForegroundColor Cyan
    $cmd = "$CC $CFLAGS -Itest -o $Tests $CoreSrc $TestSrcs"
    Write-Host "  $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    Write-Host "  Built: $Tests" -ForegroundColor Green
}

function Build-Simulator {
    Ensure-BuildDir
    Write-Host "Building simulator..." -ForegroundColor Cyan
    $cmd = "$CC $CFLAGS -o $Simulator $CoreSrc $SimSrc"
    Write-Host "  $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    Write-Host "  Built: $Simulator" -ForegroundColor Green
}

function Run-Clean {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
    Get-ChildItem -Recurse -Include "*.gcno","*.gcda","*.gcov" | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "  Clean complete." -ForegroundColor Green
}

switch ($Target.ToLower()) {
    "build" {
        Build-Demo
    }
    "all" {
        Build-Demo
        Build-Tests
        Build-Simulator
    }
    "run" {
        Build-Demo
        Write-Host ""
        & ".\$Demo"
    }
    "test" {
        Build-Tests
        Write-Host ""
        & ".\$Tests"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`nSome tests FAILED." -ForegroundColor Red
            exit 1
        }
    }
    "sim" {
        Build-Simulator
        Write-Host ""
        & ".\$Simulator"
    }
    "clean" {
        Run-Clean
    }
    "coverage" {
        Ensure-BuildDir
        $CovExe = "$BuildDir/flightguard_tests_cov.exe"
        Write-Host "Building with coverage instrumentation..." -ForegroundColor Cyan
        $cmd = "$CC $CFLAGS -Itest --coverage -o $CovExe $CoreSrc $TestSrcs"
        Invoke-Expression $cmd
        Write-Host "Running tests with coverage..." -ForegroundColor Cyan
        & ".\$CovExe"
        Write-Host "Generating coverage data..." -ForegroundColor Cyan
        Invoke-Expression "gcov $CoreSrc"
        if (-not (Test-Path "$BuildDir/coverage")) {
            New-Item -ItemType Directory -Path "$BuildDir/coverage" | Out-Null
        }
        Move-Item -Force "*.gcov" "$BuildDir/coverage/" -ErrorAction SilentlyContinue
        Move-Item -Force "*.gcno" "$BuildDir/coverage/" -ErrorAction SilentlyContinue
        Move-Item -Force "*.gcda" "$BuildDir/coverage/" -ErrorAction SilentlyContinue
        Write-Host "Coverage data in $BuildDir/coverage/" -ForegroundColor Green
    }
    default {
        Write-Host "Usage: .\build.ps1 [build|all|run|test|sim|clean|coverage]"
    }
}
