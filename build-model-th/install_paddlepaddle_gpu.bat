@echo off
REM ===================================================
REM Task 4: Install PaddlePaddle GPU - RTX 5090 Optimized
REM ===================================================

echo 🎯 Task 4: Install PaddlePaddle GPU - RTX 5090 Optimized
echo ==========================================================

cd /d "%~dp0"

REM ตรวจสอบว่ามี Python หรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ไม่พบ Python ในระบบ
    echo กรุณาติดตั้ง Python ก่อน
    pause
    exit /b 1
)

echo ✅ พบ Python
python --version

REM รันสคริปต์ Python
echo.
echo 🚀 กำลังรันสคริปต์ติดตั้ง...
echo.

python install_paddlepaddle_gpu.py

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Task 4 เสร็จสมบูรณ์!
    echo ✅ PaddlePaddle GPU พร้อมใช้งานกับ RTX 5090
) else (
    echo.
    echo ❌ Task 4 ไม่สำเร็จ
    echo ตรวจสอบข้อผิดพลาดด้านบน
)

echo.
pause
