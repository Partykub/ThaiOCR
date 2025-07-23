@echo off
echo =====================================
echo PaddlePaddle RTX 5090 Build Process
echo =====================================

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d "%~dp0\Paddle\build"

echo Starting PaddlePaddle build for RTX 5090...
echo This will take 2-4 hours depending on your system
echo RAM usage: up to 16GB during compilation
echo Disk space: up to 20GB temporary files
echo.

REM Start build with parallel compilation
echo Building with maximum CPU cores...
cmake --build . --config Release --parallel

if %ERRORLEVEL% == 0 (
    echo.
    echo Build completed successfully!
    echo Checking for wheel file...
    
    if exist "python\dist\*.whl" (
        echo Python wheel file created
        dir "python\dist\*.whl"
        echo.
        echo Ready for Phase 5: Installation and Testing
        echo Next: Run phase5_install_and_test.bat
    ) else (
        echo Wheel file not found in python\dist\
        echo Searching for wheel files...
        dir /s "*.whl"
    )
) else (
    echo.
    echo Build failed!
    echo Check error messages above
    echo Common build issues:
    echo   - Out of memory (need 16GB+ RAM)
    echo   - Out of disk space (need 20GB+)
    echo   - CUDA compilation errors
    echo   - Missing dependencies
    echo.
)

pause
