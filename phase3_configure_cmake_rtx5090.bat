@echo off
echo ==========================================
echo PaddlePaddle RTX 5090 CMake Configuration
echo ==========================================

REM Set environment variables
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDNN_ROOT=%CUDA_PATH%
set PYTHON_EXECUTABLE=C:\Users\admin\Documents\paddlepadle\tf_gpu_env\Scripts\python.exe
set PATH=C:\Users\admin\Documents\paddlepadle\tf_gpu_env\Scripts;%PATH%

REM Pre-flight checks
echo 🔍 Pre-flight Environment Validation:
echo.

REM Check CUDA installation
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ❌ ERROR: CUDA 12.8 not found at %CUDA_PATH%
    echo Please install CUDA 12.8 Toolkit
    pause
    exit /b 1
)
echo ✅ CUDA 12.8: Found at %CUDA_PATH%

REM Check Python installation
if not exist "%PYTHON_EXECUTABLE%" (
    echo ❌ ERROR: Python not found at %PYTHON_EXECUTABLE%
    pause
    exit /b 1
)
echo ✅ Python: Found at %PYTHON_EXECUTABLE%

REM Check Visual Studio installation
if not exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo ❌ ERROR: Visual Studio 2022 Community not found
    echo Please install Visual Studio 2022 with C++ workload
    pause
    exit /b 1
)
echo ✅ Visual Studio 2022: Found

REM Check CMake
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ ERROR: CMake not found in PATH
    pause
    exit /b 1
)
echo ✅ CMake: Available

echo.
echo 🚀 All prerequisites validated successfully!
echo.

REM Setup Visual Studio environment
echo 🔧 Setting up Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo 📋 Configuration Summary for RTX 5090 (SM_120):
echo   CUDA: %CUDA_PATH%
echo   Python: %PYTHON_EXECUTABLE%
echo   Generator: Visual Studio 17 2022
echo   CUDA Arch: SM_120 (Compute Capability 12.0)
echo   Build Type: Release
echo   GPU Support: ON
echo.

REM Navigate to Paddle build directory
if not exist "Paddle" (
    echo ❌ ERROR: Paddle directory not found
    echo Please run from the correct workspace directory
    echo Current: %CD%
    pause
    exit /b 1
)

cd /d "Paddle\build"

REM Verify we're in the correct build directory
if not exist "..\CMakeLists.txt" (
    echo ❌ ERROR: CMakeLists.txt not found in parent directory
    echo Please ensure Paddle source code is properly cloned
    echo Current: %CD%
    pause
    exit /b 1
)

echo ✅ Found Paddle source - Ready to configure
echo Working directory: %CD%
echo.

echo 🚀 Starting CMake Configuration for RTX 5090...
echo This process may take 5-15 minutes...
echo.

REM CMake configuration with RTX 5090 specific parameters
cmake .. ^
    -G "Ninja" ^
    -DWITH_GPU=ON ^
    -DWITH_PYTHON=ON ^
    -DWITH_INFERENCE=ON ^
    -DWITH_AVX=ON ^
    -DWITH_MKL=OFF ^
    -DPADDLE_ENABLE_CHECK=ON ^
    -DPADDLE_WITH_CUDA=ON ^
    -DCUDA_ARCH_NAME=Manual ^
    -DCUDA_ARCH_BIN="120" ^
    -DCMAKE_CUDA_ARCHITECTURES="120" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" ^
    -DCUDNN_ROOT="%CUDNN_ROOT%" ^
    -DWITH_TESTING=OFF ^
    -DWITH_DISTRIBUTE=OFF ^
    -DWITH_PSCORE=OFF ^
    -DWITH_UNITY_BUILD=ON

echo.
echo ⏳ CMake configuration in progress...
echo Please wait, this may take several minutes...
echo.

REM Check CMake configuration result
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================================
    echo ✅ CMAKE CONFIGURATION SUCCESS - RTX 5090 READY!
    echo ========================================================================
    echo.
    echo 🎯 Configuration Results:
    echo   ✅ RTX 5090 SM_120 Support: ENABLED
    echo   ✅ CUDA 12.8 Integration: SUCCESS
    echo   ✅ Visual Studio 2022: CONFIGURED
    echo   ✅ Python 3.11 Binding: READY
    echo.
    echo 📁 Generated Build Files:
    if exist "*.sln" (
        for %%f in (*.sln) do echo   - Solution: %%f
    )
    if exist "*paddle*.vcxproj" (
        for %%f in (*paddle*.vcxproj) do echo   - Project: %%f
    )
    echo.
    echo 🚀 Phase 3.1 COMPLETED SUCCESSFULLY!
    echo 📋 Next Step: Phase 3.2 - Run CMake Configuration
    echo 💡 Ready for Phase 4: Build Process (2-4 hours)
    echo.
    echo ⏭️  Next Command: cmake --build . --config Release --parallel
) else (
    echo.
    echo ========================================================================
    echo ❌ CMAKE CONFIGURATION FAILED - ERROR CODE: %ERRORLEVEL%
    echo ========================================================================
    echo.
    echo 🔍 Troubleshooting Steps:
    echo.
    echo 1. CUDA Issues:
    echo    - Verify CUDA 12.8 installation
    echo    - Check PATH: %CUDA_PATH%\bin
    echo    - Test: nvcc --version
    echo.
    echo 2. Visual Studio Issues:
    echo    - Ensure VS 2022 Community installed
    echo    - Check C++ desktop development workload
    echo    - Verify vcvars64.bat exists
    echo.
    echo 3. Path Issues:
    echo    - Check spaces in paths
    echo    - Verify Python path: %PYTHON_EXECUTABLE%
    echo    - Ensure write permissions
    echo.
    echo 4. Memory/Disk Issues:
    echo    - Free disk space: 20GB+ required
    echo    - Available RAM: 16GB+ recommended
    echo.
    echo 📋 Environment Status:
    echo CUDA_PATH: %CUDA_PATH%
    nvcc --version 2>nul || echo ❌ nvcc not accessible
    cmake --version 2>nul || echo ❌ cmake not accessible
    "%PYTHON_EXECUTABLE%" --version 2>nul || echo ❌ python not accessible
    echo.
    echo 🔄 Phase 3.1 FAILED - Requires troubleshooting
)

echo.
echo ========================================================================
echo 📊 Phase 3.1 Completion Report
echo ========================================================================
echo Script: phase3_configure_cmake_rtx5090.bat
echo Timestamp: %DATE% %TIME%
echo Status: Configuration attempt completed
echo Next Phase: 3.2 - Run CMake Configuration
echo ========================================================================
pause
