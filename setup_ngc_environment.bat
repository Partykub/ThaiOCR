@echo off
REM Setup NGC Thai OCR Environment - One-click setup
echo 🚀 NGC Thai OCR Environment Setup
echo ====================================
echo This will set up the complete NGC environment for RTX 5090
echo.
pause

REM Run the Python setup script
python setup_ngc_environment.py

echo.
echo ✅ Setup completed! 
echo 📝 You can now use the NGC container with RTX 5090 support
echo.
pause
