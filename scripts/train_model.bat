@echo off
REM Thai CRNN Training Script for RTX 5090
REM ======================================

echo Starting Thai CRNN Training...
echo.

REM Activate virtual environment
call tf_gpu_env\Scripts\activate.bat

REM Navigate to project directory
cd /d "c:\Users\admin\Documents\paddlepadle"

REM Run training
python src\training\train_thai_crnn_clean.py

echo.
echo Training completed!
pause
