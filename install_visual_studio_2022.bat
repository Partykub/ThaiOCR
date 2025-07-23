@echo off
echo ==========================================
echo Visual Studio 2022 Community Installer
echo ==========================================
echo.
echo This script will download and install Visual Studio 2022 Community
echo with C++ development tools required for PaddlePaddle RTX 5090 build.
echo.
echo Components to be installed:
echo - Desktop development with C++ workload
echo - MSVC v143 compiler toolset
echo - Windows 10/11 SDK
echo - CMake tools for Visual Studio
echo.

set VS_INSTALLER_URL=https://aka.ms/vs/17/release/vs_community.exe
set VS_INSTALLER=%TEMP%\vs_community.exe

echo Downloading Visual Studio 2022 Community installer...
powershell -Command "Invoke-WebRequest -Uri '%VS_INSTALLER_URL%' -OutFile '%VS_INSTALLER%'"

if not exist "%VS_INSTALLER%" (
    echo ERROR: Failed to download Visual Studio installer
    echo Please download manually from: https://visualstudio.microsoft.com/downloads/
    pause
    exit /b 1
)

echo.
echo Starting Visual Studio 2022 Community installation...
echo This may take 20-60 minutes depending on your internet speed.
echo.

"%VS_INSTALLER%" ^
    --add Microsoft.VisualStudio.Workload.NativeDesktop ^
    --add Microsoft.VisualStudio.Workload.Python ^
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    --add Microsoft.VisualStudio.Component.Windows10SDK.20348 ^
    --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
    --includeRecommended ^
    --quiet ^
    --wait

if %ERRORLEVEL% == 0 (
    echo.
    echo ✅ Visual Studio 2022 Community installed successfully!
    echo.
    echo Next steps:
    echo 1. Restart your computer (recommended)
    echo 2. Run phase1_environment_setup.py again to verify installation
    echo 3. Continue with Phase 2 (Clone PaddlePaddle source)
    echo.
) else (
    echo.
    echo ❌ Visual Studio installation failed or was cancelled
    echo Error code: %ERRORLEVEL%
    echo.
    echo Manual installation steps:
    echo 1. Go to: https://visualstudio.microsoft.com/downloads/
    echo 2. Download Visual Studio 2022 Community (free)
    echo 3. During installation, select "Desktop development with C++"
    echo 4. Ensure MSVC v143 compiler toolset is included
    echo.
)

echo.
echo Cleaning up installer file...
del "%VS_INSTALLER%" 2>nul

echo.
pause
