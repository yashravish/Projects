@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo Compiling vector_add.cu...
nvcc -o vector_add vector_add.cu
if %ERRORLEVEL% EQU 0 (
    echo Running vector_add...
    vector_add.exe
) else (
    echo Compilation failed!
)
echo.
echo Compiling audio_gain.cu...
nvcc -o audio_gain audio_gain.cu
if %ERRORLEVEL% EQU 0 (
    echo Running audio_gain...
    audio_gain.exe
) else (
    echo Compilation failed!
)
echo.
echo Compiling audio_mixer.cu...
nvcc -o audio_mixer audio_mixer.cu
if %ERRORLEVEL% EQU 0 (
    echo Running audio_mixer...
    audio_mixer.exe
) else (
    echo Compilation failed!
)
echo.
echo Compiling simple_filter.cu...
nvcc -o simple_filter simple_filter.cu
if %ERRORLEVEL% EQU 0 (
    echo Running simple_filter...
    simple_filter.exe
) else (
    echo Compilation failed!
)
