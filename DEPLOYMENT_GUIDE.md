# NGC Thai OCR - Deployment Guide

## 🚀 Quick Deployment (Recommended)

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd paddlepadle
```

### Step 2: One-Click Setup
```bash
# Windows
setup_ngc_environment.bat

# Linux/macOS
python setup_ngc_environment.py
```

### Step 3: Verify Installation
```bash
python verify_ngc_environment.py
```

## 🐳 Manual Docker Deployment

### Prerequisites
- Docker Desktop installed
- NVIDIA Container Toolkit configured
- RTX 5090 or compatible GPU

### Commands
```bash
# Pull NGC image
docker pull nvcr.io/nvidia/paddlepaddle:24.12-py3

# Start with docker-compose
docker-compose -f docker-compose.ngc.yml up -d

# Connect to container
docker exec -it thai-ocr-training-ngc bash
```

## 🔧 Container Environment Setup

### Inside Container (Automated by setup script)
```bash
# System dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1

# PaddlePaddle compatibility
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install paddlepaddle-gpu==2.6.2

# PaddleOCR without conflicts  
pip uninstall -y paddleocr paddlex
pip install paddleocr==2.7.0.3

# Fixed versions for compatibility
pip install numpy==1.26.4 opencv-python-headless==4.10.0.84
pip install tensorflow==2.15.0 keras==2.15.0
```

## ✅ Verification Checklist

After deployment, verify:
- [ ] Container starts successfully
- [ ] RTX 5090 detected without SM_120 warnings
- [ ] PaddlePaddle 2.6.2 with CUDA support
- [ ] PaddleOCR 2.7.0.3 working
- [ ] No PaddleX conflicts
- [ ] GPU memory allocation working

## 🎯 Usage Examples

### Basic OCR Test
```python
from paddleocr import PaddleOCR

# Initialize OCR (should work without errors)
ocr = PaddleOCR(use_gpu=True, lang='th', show_log=False)

# Test with image
result = ocr.ocr('path/to/thai_image.jpg')
print(result)
```

### GPU Verification
```python
import paddle
print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")
```

## 🔍 Troubleshooting

### Common Issues & Solutions

1. **Container won't start**
   - Check Docker Desktop is running
   - Verify NVIDIA Container Toolkit installed
   - Run: `docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi`

2. **SM_120 warnings**
   - Should be resolved with NGC container
   - If persist, verify using correct image: `nvcr.io/nvidia/paddlepaddle:24.12-py3`

3. **PaddleOCR import errors**
   - Uninstall PaddleX: `pip uninstall paddlex`
   - Reinstall specific version: `pip install paddleocr==2.7.0.3`

4. **NumPy version conflicts**
   - Downgrade NumPy: `pip install numpy==1.26.4`

### Check Container Status
```bash
# View running containers
docker ps

# Check container logs
docker logs thai-ocr-training-ngc

# Container resource usage
docker stats thai-ocr-training-ngc
```

## 📊 Performance Expectations

### RTX 5090 Performance
- **Inference**: 0.1-0.3 seconds per image
- **GPU Memory**: 4-8GB during training
- **Training Speed**: 2-5 seconds per batch
- **Accuracy**: 85-95% on clear Thai text

### Container Resources
- **RAM**: 8-16GB recommended
- **Storage**: 10-20GB for models and data
- **GPU**: RTX 5090 24GB VRAM

## 🎉 Success Indicators

You'll know deployment is successful when:
1. ✅ Container starts without errors
2. ✅ `verify_ngc_environment.py` passes all tests
3. ✅ PaddleOCR initializes with `use_gpu=True`
4. ✅ No CUDA warnings in logs
5. ✅ Thai text recognition works correctly

## 📞 Support

If deployment fails:
1. Run `verify_ngc_environment.py` for detailed diagnostics
2. Check `ngc_verification_report.json` for test results
3. Review container logs: `docker logs thai-ocr-training-ngc`
4. Ensure RTX 5090 drivers are up to date
