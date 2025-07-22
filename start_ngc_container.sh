#!/bin/bash
# Start Thai OCR NGC Container
echo "🚀 Starting Thai OCR NGC Container..."
docker start thai-ocr-training-ngc || docker-compose -f docker-compose.ngc.yml up -d
echo "✅ Container started: thai-ocr-training-ngc"
echo "📝 Connect with: docker exec -it thai-ocr-training-ngc bash"
