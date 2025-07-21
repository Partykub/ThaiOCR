@echo off
REM ===================================================
REM สคริปต์สำหรับสร้าง Thai OCR Dataset ทั้งหมด
REM ===================================================

echo 🎯 สร้าง Thai OCR Dataset
echo ===============================================

cd /d "%~dp0\.."

REM เช็คว่ามีไฟล์จำเป็นหรือไม่
if not exist "thai-letters\thai_text_generator.py" (
    echo ❌ ไม่พบไฟล์ thai_text_generator.py
    pause
    exit /b 1
)

if not exist "thai-letters\thai_corpus.txt" (
    echo ❌ ไม่พบไฟล์ thai_corpus.txt
    pause
    exit /b 1
)

echo ✅ พบไฟล์จำเป็นครบถ้วน

REM สร้าง Dataset ตัวอย่าง (500 ภาพ)
echo.
echo 🚀 สร้าง Dataset ตัวอย่าง (500 ภาพ)...
echo คำสั่ง: python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 500 -o thai-letters/generated_text_samples
echo --------------------------------------------------
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 500 -o thai-letters/generated_text_samples

if %errorlevel% equ 0 (
    echo ✅ สร้าง Dataset ตัวอย่าง สำเร็จ!
) else (
    echo ❌ สร้าง Dataset ตัวอย่าง ล้มเหลว!
    pause
    exit /b 1
)

REM สร้าง Dataset ครบชุด (9,672 ภาพ)
echo.
echo 🚀 สร้าง Dataset ครบชุด (9,672 ภาพ)...
echo คำสั่ง: python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 9672 -o thai-letters/full_corpus_dataset
echo --------------------------------------------------
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 9672 -o thai-letters/full_corpus_dataset

if %errorlevel% equ 0 (
    echo ✅ สร้าง Dataset ครบชุด สำเร็จ!
) else (
    echo ❌ สร้าง Dataset ครบชุด ล้มเหลว!
    pause
    exit /b 1
)

echo.
echo 🎉 การสร้าง dataset เสร็จสิ้น!
echo 📁 ตรวจสอบโฟลเดอร์:
echo    - thai-letters\thai_ocr_dataset\ (5,000 ภาพ - เดิม)
echo    - thai-letters\generated_text_samples\ (500 ภาพ)
echo    - thai-letters\full_corpus_dataset\ (9,672 ภาพ)
echo.
echo 📊 รวมทั้งหมด: 15,172 ภาพ
echo.

pause
