# Requirements for PaddleOCR Thai Language Support

## Basic Installation Requirements

### Core PaddlePaddle and PaddleOCR
```
# Core framework (use GPU version for RTX 5090)
paddlepaddle-gpu>=2.6.0
paddleocr>=2.7.0

# Alternative CPU version (fallback)
# paddlepaddle>=2.6.0
```

### Image Processing Libraries
```
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.21.0
scikit-image>=0.19.0
```

### Text Processing and OCR Utilities
```
imgaug>=0.4.0
lmdb>=1.3.0
tqdm>=4.64.0
matplotlib>=3.5.0
rapidfuzz>=2.0.0
shapely>=1.8.0
lanms-neo>=1.0.2
```

### Configuration and Data Handling
```
pyyaml>=6.0
protobuf>=3.20.0
wheel>=0.37.0
```

### Training and Development Tools
```
visualdl>=2.4.0
tensorboard>=2.10.0
sklearn>=1.1.0
pandas>=1.5.0
seaborn>=0.11.0
```

### Data Augmentation (Optional)
```
albumentations>=1.3.0
fonttools>=4.33.0
```

### Integration with Existing CRNN Model
```
# For compatibility with thai-license-plate-recognition-CRNN
tensorflow>=2.3.0,<2.16.0
keras>=2.3.0
h5py>=3.1.0
```

### Web API Development (Optional)
```
flask>=2.0.0
requests>=2.25.0
gunicorn>=20.1.0
```

### Jupyter Notebook Support (Optional)
```
jupyter>=1.0.0
ipywidgets>=7.6.0
```

## Development Requirements

### Code Quality and Testing
```
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
isort>=5.0.0
```

### Documentation
```
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
myst-parser>=0.17.0
```

## Installation Notes

### For RTX 5090 Users
1. **CUDA 12.6** and **cuDNN 9.0** required
2. May need to compile PaddlePaddle from source with SM 120 support
3. Fallback to CPU version if GPU compilation fails

### For Windows Users
```bash
# Install Visual Studio Build Tools first
# Then install packages
pip install -r requirements.txt
```

### For Linux Users
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install python3-dev libgl1-mesa-glx libglib2.0-0

# Then install Python packages
pip install -r requirements.txt
```

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10-3.11 | 64-bit required |
| CUDA | 12.6 | For RTX 5090 support |
| cuDNN | 9.0 | Compatible with CUDA 12.6 |
| PaddlePaddle | >=2.6.0 | GPU version recommended |
| PaddleOCR | >=2.7.0 | Latest stable version |
| OpenCV | >=4.8.0 | For image processing |
| NumPy | >=1.21.0 | Core dependency |

## Hardware Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB system memory
- **Storage**: 5GB free space
- **GPU**: Optional but recommended

### Recommended Requirements (for RTX 5090)
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ SSD free space
- **GPU**: RTX 5090 with 24GB VRAM
- **CUDA**: 12.6 with compute capability 12.0

## Quick Installation Commands

### Standard Installation
```bash
# Clone repository
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Install requirements
pip install -r requirements.txt

# Install Thai language requirements
pip install -r thai-letters/requirements.txt
```

### GPU Installation (RTX 5090)
```bash
# Install CUDA 12.6 and cuDNN 9.0 first
# Then install GPU version
pip install paddlepaddle-gpu>=2.6.0
pip install paddleocr>=2.7.0
pip install -r thai-letters/requirements.txt
```

### CPU-only Installation
```bash
pip install paddlepaddle>=2.6.0
pip install paddleocr>=2.7.0
pip install -r thai-letters/requirements.txt
```

### Development Installation
```bash
# Install all requirements including development tools
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Troubleshooting Installation

### Common Issues
1. **GPU not detected**: Check CUDA installation and compatibility
2. **Module import errors**: Verify Python environment and package versions
3. **Memory errors**: Reduce batch size or use CPU version
4. **Build failures**: Ensure Visual Studio Build Tools are installed (Windows)

### Verification Commands
```python
# Test basic installation
import paddle
import paddleocr
print("Installation successful!")

# Test GPU support
print("CUDA available:", paddle.device.is_compiled_with_cuda())
print("GPU count:", paddle.device.cuda.device_count())

# Test OCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=True)
print("OCR initialized successfully!")
```

For detailed troubleshooting, see [Troubleshooting Guide](troubleshooting.md).
