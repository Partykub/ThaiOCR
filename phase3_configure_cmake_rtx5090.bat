@echo off
echo ==========================================
echo PaddlePaddle RTX 5090 CMake Configuration
echo ==========================================

REM Set environment variables
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDNN_ROOT=%CUDA_PATH%
set PYTHON_EXECUTABLE=C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Configuration for RTX 5090 (SM_120):
echo   CUDA: %CUDA_PATH%
echo   Python: %PYTHON_EXECUTABLE%
echo   Generator: Visual Studio 17 2022
echo   CUDA Arch: SM_120 (Compute Capability 12.0)
echo.

cd /d "%~dp0\Paddle\build"

echo Running CMake configuration...
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DWITH_GPU=ON ^
    -DWITH_PYTHON=ON ^
    -DWITH_INFERENCE=ON ^
    -DWITH_AVX=ON ^
    -DWITH_MKL=OFF ^
    -DPADDLE_ENABLE_CHECK=ON ^
    -DPADDLE_WITH_CUDA=ON ^
    -DCUDA_ARCH_NAME=Manual ^
    -DCUDA_ARCH_BIN="120" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" ^
    -DCUDNN_ROOT="%CUDNN_ROOT%" ^
    -DWITH_TESTING=OFF ^
    -DWITH_DISTRIBUTE=OFF

if %ERRORLEVEL% == 0 (
    echo.
    echo CMake configuration successful!
    echo Next: Run phase4_build_paddle_rtx5090.bat to start compilation
    echo Expected build time: 2-4 hours
    echo.
) else (
    echo.
    echo CMake configuration failed!
    echo Check error messages above
    echo Common issues:
    echo   - CUDA path incorrect
    echo   - Visual Studio not found
    echo   - Python path incorrect
    echo.
)

pause
