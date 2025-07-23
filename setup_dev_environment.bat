@echo off
echo ==========================================
echo PaddlePaddle RTX 5090 Developer Environment
echo ==========================================

REM Find Visual Studio installation
set VS_PATH=
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
    set VS_VERSION=2022 Community
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
    set VS_VERSION=2022 Professional
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat
    set VS_VERSION=2022 Enterprise
)

if "%VS_PATH%"=="" (
    echo ❌ Visual Studio 2022 not found!
    echo Please install Visual Studio 2022 with C++ tools first.
    echo Run: install_visual_studio_2022.bat
    pause
    exit /b 1
)

echo ✅ Found Visual Studio %VS_VERSION%
echo 🔧 Setting up developer environment...

REM Setup Visual Studio environment
call "%VS_PATH%"

REM Verify tools
echo.
echo 🔍 Verifying development tools:

where cl >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ MSVC Compiler: Available
    cl 2>&1 | findstr "Version"
) else (
    echo ❌ MSVC Compiler: Not found
)

where cmake >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ CMake: Available
    cmake --version | findstr "cmake version"
) else (
    echo ❌ CMake: Not found
)

python -c "import ninja; print('✅ Ninja: Available')" 2>nul || echo ❌ Ninja: Not found

where nvcc >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ CUDA: Available
    nvcc --version | findstr "release"
) else (
    echo ❌ CUDA: Not found
)

echo.
echo 🎯 Environment ready for PaddlePaddle RTX 5090 build!
echo.
echo Next steps:
echo 1. Run: git clone https://github.com/PaddlePaddle/Paddle.git
echo 2. Run: configure_cmake_rtx5090.bat
echo 3. Run: build_paddle_rtx5090.bat
echo.

REM Keep window open
cmd /k
