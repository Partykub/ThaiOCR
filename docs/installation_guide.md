# Installation Guide for PaddleOCR Thai

## Prerequisites

### System Requirements
- **OS**: Windows 11 (recommended), Windows 10, or Linux
- **Python**: 3.10 or 3.11 (64-bit)
- **GPU**: RTX 5090 or compatible NVIDIA GPU with Compute Capability 12.0
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Software Dependencies
1. **Visual Studio 2019** with C/C++ workload
2. **Git** for version control
3. **CMake** 3.17 or higher

## Step 1: CUDA and cuDNN Installation

### Download and Install CUDA 12.6
```bash
# Download from NVIDIA website
# https://developer.nvidia.com/cuda-12-6-0-download-archive

# Install to default location:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### Download and Install cuDNN 9.0
```bash
# Download cuDNN 9.0 for CUDA 12.x
# Extract to CUDA installation directory:
# - Copy bin/* to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
# - Copy include/* to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include  
# - Copy lib/* to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib
```

### Set Environment Variables
```batch
set CUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%CUDA_TOOLKIT_ROOT_DIR%\libnvvp;%PATH%
```

## Step 2: Compile PaddlePaddle for RTX 5090

### Clone PaddlePaddle Repository
```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout develop
```

### Create Build Directory
```bash
mkdir build
cd build
```

### Configure CMake (Critical for RTX 5090)
```bash
# Using Ninja (recommended)
cmake .. -GNinja ^
  -DWITH_GPU=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DWITH_UNITY_BUILD=ON ^
  -DCUDA_ARCH_NAME=Manual ^
  -DCUDA_ARCH_BIN="120"

# Or using Visual Studio
cmake .. -G "Visual Studio 16 2019" -A x64 ^
  -DWITH_GPU=ON -DCMAKE_BUILD_TYPE=Release -DWITH_UNITY_BUILD=ON ^
  -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="120"
```

**Important**: `-DCUDA_ARCH_BIN="120"` is crucial for RTX 5090 support (Compute Capability 12.0)

### Build PaddlePaddle
```bash
# With Ninja
ninja -j8

# With Visual Studio
# Open paddle.sln in VS 2019, select x64 Release, then Build Solution
```

### Install Built Package
```bash
cd python\dist
pip install paddlepaddle_gpu-*.whl
```

## Step 3: Install PaddleOCR

```bash
pip install paddleocr
```

## Step 4: Install Project Dependencies

```bash
# Navigate to project directory
cd c:\Users\admin\Documents\paddlepadle\thai-letters

# Install requirements
pip install -r requirements.txt
```

## Step 5: Verify Installation

### Test PaddlePaddle GPU Support
```python
import paddle
print("CUDA compiled:", paddle.device.is_compiled_with_cuda())
print("GPU count:", paddle.device.cuda.device_count())
print("Device:", paddle.device.get_device())

# Run comprehensive check
paddle.utils.run_check()
```

Expected output:
```
CUDA compiled: True
GPU count: 1
Device: gpu:0
PaddlePaddle is installed successfully!
```

### Test PaddleOCR
```python
from paddleocr import PaddleOCR
import cv2

# Test with existing dataset
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
result = ocr.ocr('c:/Users/admin/Documents/paddlepadle/thai-letters/thai_ocr_dataset/images/000001.jpg')
print("OCR working:", len(result) > 0)
```

## Troubleshooting

### Common Issues

#### 1. "no kernel image is available" Error
```bash
# Solution: Ensure CUDA_ARCH_BIN="120" is set during compilation
# Verify GPU compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### 2. CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Ensure consistency between toolkit and driver
```

#### 3. Build Fails
```bash
# Try CPU version as fallback
pip install paddlepaddle

# Or use pre-built wheel if available
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

### Alternative Installation (If Build Fails)

```bash
# Option 1: Use CUDA 12.3 instead of 12.6
# Download CUDA 12.3 and modify CMAKE commands accordingly

# Option 2: Use CPU version for development
pip install paddlepaddle
# Note: Training will be slower but functional

# Option 3: Use WSL2 with Linux
# May have better CUDA support for newer GPUs
```

## Performance Optimization

### GPU Memory Settings
```python
import paddle
paddle.device.set_device('gpu:0')
paddle.device.cuda.set_device(0)

# Set memory allocation strategy
paddle.fluid.memory_optimize()
```

### Environment Variables for Optimization
```batch
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
```

## Next Steps

After successful installation:
1. [Training Guide](training_guide.md) - Train Thai recognition model
2. [API Reference](api_reference.md) - Integrate into applications
3. [Troubleshooting](troubleshooting.md) - Solve common issues
