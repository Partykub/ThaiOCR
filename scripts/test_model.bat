@echo off
REM Thai CRNN Model Testing Script
REM ==============================

echo Starting Thai CRNN Model Testing...
echo.

REM Activate virtual environment
call tf_gpu_env\Scripts\activate.bat

REM Navigate to project directory
cd /d "c:\Users\admin\Documents\paddlepadle"

REM Run testing
python src\testing\test_thai_crnn_clean.py

echo.
echo Testing completed!
pause
