@echo off
REM Clean Old Build Files Script
REM ============================

echo Cleaning old build-model-th files...
echo.

REM Remove old files and empty directories
if exist "build-model-th\*.bat" del "build-model-th\*.bat"
if exist "build-model-th\*.jpg" del "build-model-th\*.jpg" 
if exist "build-model-th\*.h5" del "build-model-th\*.h5"
if exist "build-model-th\*.md" del "build-model-th\*.md"
if exist "build-model-th\checkpoints" rmdir /s /q "build-model-th\checkpoints" 2>nul

echo ✅ Cleaned old files
echo.
echo Project structure is now clean and organized!
echo.

REM Show new structure
echo 📁 New Project Structure:
echo ├── src/              # Source code
echo ├── models/           # Trained models  
echo ├── scripts/          # Batch scripts
echo ├── configs/          # Configuration
echo ├── logs/             # Training logs
echo └── archive/          # Old files

pause
