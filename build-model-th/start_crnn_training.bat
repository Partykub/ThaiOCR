@echo off
REM Enhanced CRNN Training Script for Windows
REM Optimized for RTX 5090 GPU

echo ===============================================================
echo 🇹🇭 Thai CRNN Training System - RTX 5090 Optimized
echo ===============================================================
echo.

REM Set RTX 5090 environment variables
echo 🎮 Configuring RTX 5090 environment...
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_MEMORY_ALLOW_GROWTH=true

REM Set CUDA paths
set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%PATH%

echo ✅ Environment configured for RTX 5090
echo.

REM Check if Python is available
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)
echo ✅ Python found
echo.

REM Change to the project directory
echo 📂 Navigating to project directory...
cd /d "%~dp0"
if not exist "start_crnn_training.py" (
    echo ❌ Training script not found in current directory
    echo Current directory: %CD%
    pause
    exit /b 1
)
echo ✅ Training script found
echo.

REM Check GPU status
echo 🔍 Checking GPU status...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ NVIDIA-SMI not found. GPU may not be available.
    echo Continuing with CPU training...
) else (
    echo ✅ GPU detected
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
)
echo.

REM Start training
echo 🚀 Starting CRNN training...
echo.
echo Training started at: %date% %time%
echo.

python start_crnn_training.py

REM Check training result
if %errorlevel% equ 0 (
    echo.
    echo ===============================================================
    echo 🎉 Training completed successfully!
    echo ===============================================================
    echo.
    echo 📊 Training Results:
    echo   ✅ Model saved to: ./checkpoints/
    echo   📝 Logs saved to: crnn_training.log
    echo   📈 History saved to: ./checkpoints/training_history.json
    echo.
    echo Training completed at: %date% %time%
) else (
    echo.
    echo ===============================================================
    echo ❌ Training failed or was interrupted
    echo ===============================================================
    echo.
    echo Please check crnn_training.log for details
)

echo.
echo Press any key to exit...
pause >nul
