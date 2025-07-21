@echo off
REM Thai OCR Dependencies Installation Script
REM For RTX 5090 + PaddleOCR + Thai language support

echo ========================================
echo 🚀 Thai OCR Dependencies Installation
echo ========================================
echo Installing packages for RTX 5090 compatibility
echo.

REM Change to build-model-th directory
cd /d "%~dp0"
echo 📂 Working directory: %CD%
echo.

REM Check Python installation
echo 🐍 Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found! Please install Python 3.10+ first.
    pause
    exit /b 1
)
echo.

REM Set environment variables for RTX 5090
echo 🔧 Setting RTX 5090 environment variables...
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
echo ✅ Environment variables set for optimal RTX 5090 performance
echo.

REM Run the Python installation script
echo 📦 Starting Python installation script...
python install_dependencies.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Installation failed with errors.
    echo 💡 Check the output above for details.
    echo.
    echo 🔧 Troubleshooting tips:
    echo    1. Make sure you have Python 3.10+ installed
    echo    2. Check internet connection
    echo    3. Try running as administrator
    echo    4. Check CUDA 12.6 installation
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 Installation completed!
echo ========================================
echo.
echo 📚 Next steps:
echo 1. Test installation: Ctrl+Shift+P ^> Tasks: Run Task ^> "Test PaddleOCR Installation"
echo 2. Check GPU: Ctrl+Shift+P ^> Tasks: Run Task ^> "Check GPU Status"
echo 3. Generate dataset: Ctrl+Shift+P ^> Tasks: Run Task ^> "Generate Thai Text Dataset"
echo.
echo 🎮 Your RTX 5090 is ready for Thai OCR training!
echo.
pause
