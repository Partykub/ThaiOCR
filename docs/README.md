# Build Model TH - Thai OCR Dependencies

## Overview

This directory contains automated installation scripts for setting up the Thai OCR development environment, specifically optimized for RTX 5090 GPU.

## Files

### 📦 `install_dependencies.py`
Comprehensive Python script that installs all required packages for Thai OCR development:

- **PaddlePaddle GPU** - Optimized for RTX 5090
- **PaddleOCR** - With Thai language support
- **Base packages** - NumPy, OpenCV, Pillow, etc.
- **Thai-specific** - pythainlp, fonttools
- **ML frameworks** - TensorFlow, Keras (CRNN compatibility)
- **Environment setup** - RTX 5090 optimization

### 🚀 `install_thai_ocr.bat`
Windows batch script for easy one-click installation:

- Sets up environment variables
- Runs the Python installation script
- Provides troubleshooting tips
- Shows next steps after installation

## Quick Start

### Method 1: VS Code Task (Recommended)
1. Press **Ctrl+Shift+P**
2. Type **"Tasks: Run Task"**
3. Select **"Install Thai OCR Dependencies"**

### Method 2: Windows Batch Script
```batch
cd build-model-th
install_thai_ocr.bat
```

### Method 3: Python Script
```bash
cd build-model-th
python install_dependencies.py
```

## What Gets Installed

### 🔥 Core PaddlePaddle Stack
- `paddlepaddle-gpu>=2.6.0` - GPU version for RTX 5090
- `paddleocr>=2.7.0` - OCR engine with Thai support

### 🐍 Python Base Packages
- `numpy>=1.21.0` - Numerical computing
- `opencv-python>=4.8.0` - Computer vision
- `pillow>=9.0.0` - Image processing
- `matplotlib>=3.5.0` - Plotting and visualization

### 🇹🇭 Thai Language Support
- `pythainlp` - Thai natural language processing
- `fonttools` - Font manipulation for Thai text
- `Wand` - ImageMagick binding for image generation
- `reportlab` - PDF generation with Thai fonts

### 🤖 Machine Learning
- `tensorflow>=2.3.0,<2.16.0` - For CRNN compatibility
- `keras>=2.3.0` - High-level neural networks
- `h5py>=3.1.0` - HDF5 file format support
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation

## RTX 5090 Optimization

The installation automatically configures these environment variables:

```bash
FLAGS_fraction_of_gpu_memory_to_use=0.8    # Use 80% of 24GB VRAM
FLAGS_conv_workspace_size_limit=512        # Optimize convolution workspace
FLAGS_cudnn_deterministic=true             # Ensure reproducible results
```

## Installation Process

### 1. Environment Check
- ✅ Python version verification (3.10+ recommended)
- ✅ GPU driver detection
- ✅ CUDA availability check

### 2. Package Installation
- 📦 Base packages (pip, wheel, setuptools)
- 🔥 PaddlePaddle GPU (with fallback to CPU)
- 🔤 PaddleOCR and dependencies
- 🇹🇭 Thai-specific packages
- 🤖 ML framework packages

### 3. Verification
- 🧪 Import tests for all major packages
- 🎮 GPU functionality verification
- 📊 Performance optimization validation

### 4. Summary Report
- ✅ Success/failure status for each component
- 💡 Troubleshooting suggestions for failures
- 📚 Next steps recommendations

## Expected Performance

### RTX 5090 Benchmarks
- **Installation time**: 3-7 minutes (depending on internet speed)
- **GPU memory usage**: 19.2GB available (80% of 24GB)
- **CUDA compute capability**: 12.0 (SM 120)
- **Inference speed**: 0.1-0.3 seconds per image

### Memory Requirements
- **RAM**: 8GB+ recommended during installation
- **Storage**: 5GB+ for all packages
- **GPU VRAM**: 24GB (RTX 5090)

## Troubleshooting

### Common Issues

#### 1. Python Not Found
```
❌ Python not found! Please install Python 3.10+ first.
```
**Solution**: Install Python 3.10 or 3.11 from python.org

#### 2. CUDA Compilation Failed
```
❌ PaddlePaddle GPU installation failed
```
**Solutions**:
- Check NVIDIA driver version (555.0+)
- Install CUDA 12.6 toolkit
- Verify cuDNN 9.0 installation
- Run `nvidia-smi` to check GPU status

#### 3. Permission Errors
```
❌ Access denied during package installation
```
**Solution**: Run Command Prompt as Administrator

#### 4. Network Issues
```
❌ Failed to download packages
```
**Solutions**:
- Check internet connection
- Try using pip with `--trusted-host` flags
- Use corporate proxy settings if needed

### Manual Fallback

If automated installation fails, install manually:

```bash
# Essential packages only
pip install numpy opencv-python pillow
pip install paddlepaddle-gpu paddleocr

# Test basic functionality
python -c "import paddle; import paddleocr; print('Basic installation OK')"
```

## Verification Commands

After installation, verify with these commands:

```python
# Test PaddlePaddle GPU
import paddle
print(f"CUDA: {paddle.device.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")

# Test PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=True, lang='th')
print("PaddleOCR Thai ready!")

# Test Thai language support
import pythainlp
print(f"PyThaiNLP: {pythainlp.__version__}")
```

## Next Steps

After successful installation:

1. **🧪 Test Installation**
   ```
   Tasks: Run Task > "Test PaddleOCR Installation"
   ```

2. **🎮 Check GPU Status**
   ```
   Tasks: Run Task > "Check GPU Status"
   ```

3. **📊 Generate Dataset**
   ```
   Tasks: Run Task > "Generate Thai Text Dataset"
   ```

4. **🚀 Start Development**
   - Create Thai OCR training data
   - Train custom recognition models
   - Deploy OCR applications

## Support

For issues or questions:
- Check [Troubleshooting Guide](../docs/troubleshooting.md)
- Review [Installation Guide](../docs/installation_guide.md)
- Consult [API Reference](../docs/api_reference.md)

---

*Optimized for RTX 5090 with CUDA 12.6 and PaddlePaddle 2.6+*
