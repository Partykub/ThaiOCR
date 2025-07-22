@echo off
REM Start Thai OCR NGC Container
echo 🚀 Starting Thai OCR NGC Container...
docker start thai-ocr-training || docker-compose -f docker-compose.ngc.yml up -d
echo ✅ Container started: thai-ocr-training
echo 📝 Connect with: docker exec -it thai-ocr-training bash
pause
