@echo off
REM Final Project Cleanup Script
REM =============================

echo 🧹 Final Project Cleanup
echo ========================
echo.

echo 🗑️  Removing unnecessary files...

REM Clean Python cache
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
if exist "src\__pycache__" rmdir /s /q "src\__pycache__" 2>nul
if exist "src\models\__pycache__" rmdir /s /q "src\models\__pycache__" 2>nul
if exist "src\training\__pycache__" rmdir /s /q "src\training\__pycache__" 2>nul
if exist "src\testing\__pycache__" rmdir /s /q "src\testing\__pycache__" 2>nul
if exist "src\utils\__pycache__" rmdir /s /q "src\utils\__pycache__" 2>nul

REM Clean temporary files
if exist "*.tmp" del "*.tmp" /q 2>nul
if exist "*.temp" del "*.temp" /q 2>nul
if exist "*.bak" del "*.bak" /q 2>nul

REM Clean logs except essential ones
if exist "logs\*.log" del "logs\*.log" /q 2>nul

echo ✅ Cleanup completed!
echo.

echo 📁 Final Project Structure:
echo ├── src/              # 🎯 Source code
echo │   ├── models/       # 🧠 Model architecture
echo │   ├── training/     # 🏋️ Training scripts  
echo │   ├── testing/      # 🧪 Testing scripts
echo │   └── utils/        # 🛠️ Utilities
echo ├── models/           # 💾 Trained models
echo ├── scripts/          # ⚡ Batch scripts
echo ├── configs/          # ⚙️ Configuration
echo ├── logs/             # 📊 Training logs
echo ├── docs/             # 📚 Documentation
echo ├── archive/          # 📦 Archived files
echo └── thai-letters/     # 📝 Dataset

echo.
echo 🎉 Project is now clean and organized!
echo 🚀 Ready for training and development!

pause
