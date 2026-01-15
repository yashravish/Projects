# Set up Visual Studio environment
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

# Function to compile and run a CUDA program
function Compile-And-Run {
    param(
        [string]$sourceFile,
        [string]$outputName
    )

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Compiling $sourceFile..." -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Run compilation in cmd environment with architecture flags
    # RTX 4050 is Ada Lovelace architecture (compute capability 8.9, use sm_89)
    $result = & cmd.exe /c "`"$vsPath`" x64 >nul 2>&1 && nvcc -arch=sm_89 -o $outputName $sourceFile 2>&1"
    if ($result) {
        Write-Host $result
    }

    if (Test-Path "$outputName.exe") {
        Write-Host "`nCompilation SUCCESS! Running $outputName..." -ForegroundColor Green
        Write-Host "----------------------------------------`n" -ForegroundColor Yellow
        & ".\$outputName.exe"
        Write-Host "`n" -ForegroundColor Yellow
    } else {
        Write-Host "Compilation FAILED for $sourceFile" -ForegroundColor Red
    }
}

# Compile and run all programs
Compile-And-Run "vector_add.cu" "vector_add"
Compile-And-Run "audio_gain.cu" "audio_gain"
Compile-And-Run "audio_mixer.cu" "audio_mixer"
Compile-And-Run "simple_filter.cu" "simple_filter"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All tests completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
