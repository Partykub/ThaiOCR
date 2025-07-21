# Development Tasks - Thai OCR Project

## Overview

This document provides a comprehensive guide to all development tasks available in the Thai OCR project. These tasks are designed to streamline the development workflow and make it easy to work with PaddleOCR, CRNN models, and RTX 5090 GPU optimization.

## How to Run Tasks

### Method 1: VS Code Command Palette
1. Press **Ctrl+Shift+P**
2. Type **"Tasks: Run Task"**
3. Select the desired task from the list

### Method 2: Build/Test Tasks
- **Ctrl+Shift+P** → **"Tasks: Run Build Task"** - For setup and building
- **Ctrl+Shift+P** → **"Tasks: Run Test Task"** - For testing and validation

### Method 3: Terminal
Run tasks directly in terminal:
```bash
# Navigate to project directory
cd c:\Users\admin\Documents\paddlepadle

# Run specific commands (see task details below)
```

## Build Tasks 🔧

### 1. Install Thai OCR Dependencies
**Purpose**: Install all required Python packages for Thai OCR functionality optimized for RTX 5090

**Command**: 
```bash
cd build-model-th && python install_dependencies.py
```

**Alternative (Windows)**:
```bash
cd build-model-th && install_thai_ocr.bat
```

**What it installs**:
- 🔥 PaddlePaddle GPU (RTX 5090 optimized)
- 🔤 PaddleOCR with Thai language support
- 🐍 Python base packages (NumPy, OpenCV, Pillow)
- 🇹🇭 Thai-specific packages (pythainlp, fonttools)
- 🤖 ML frameworks (TensorFlow, Keras for CRNN compatibility)
- 🎮 RTX 5090 environment configuration

**When to use**:
- First time project setup
- After pulling new dependencies  
- When environment is corrupted
- Setting up new development machine

**Expected output**: 
- ✅ All packages installed successfully
- 🎮 RTX 5090 environment variables configured
- 🧪 Installation verification completed
- 📊 GPU support test results

**Features**:
- 🔧 Automatic RTX 5090 optimization
- 🔄 Fallback to CPU if GPU installation fails
- 🧪 Comprehensive installation verification
- 📊 Detailed progress reporting

---

### 2. Generate Thai Text Dataset
**Purpose**: Create synthetic Thai text data for training

**Command**:
```bash
python thai-letters/thai_text_generator.py
```

**When to use**:
- Need more training data
- Expanding vocabulary coverage
- Creating specialized text samples

**Output**: Generated text files in various Thai scripts and fonts

---

### 3. Create Thai OCR Dataset
**Purpose**: Generate image-text pairs for OCR training

**Command**:
```bash
python thai-letters/create_thai_ocr_dataset.py
```

**When to use**:
- Preparing training dataset
- Creating custom image samples
- Data augmentation

**Output**: Images with corresponding labels in `thai_ocr_dataset/`

---

### 4. Install PaddlePaddle GPU
**Purpose**: Install GPU-optimized PaddlePaddle and PaddleOCR

**Command**:
```bash
pip install paddlepaddle-gpu paddleocr
```

**When to use**:
- Fresh environment setup
- GPU driver updates
- Version upgrades

**Requirements**: CUDA 12.6, cuDNN 9.0, RTX 5090

---

### 5. Start CRNN Training
**Purpose**: Begin training the CRNN model for license plate recognition

**Command**:
```bash
cd thai-license-plate-recognition-CRNN && python training.py
```

**When to use**:
- Training new models
- Fine-tuning existing models
- Experimenting with parameters

**Note**: This is a background task that may run for hours

---

### 6. Setup Environment for RTX 5090
**Purpose**: Configure environment variables for optimal RTX 5090 performance

**Command**:
```bash
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
```

**When to use**:
- Before training large models
- When facing GPU memory issues
- Optimizing performance

**Effect**: Limits GPU memory usage and optimizes CUDA operations

## Test Tasks 🧪

### 1. Test PaddleOCR Installation
**Purpose**: Verify PaddleOCR and CUDA are working correctly

**Command**:
```bash
python -c "import paddle; print('CUDA:', paddle.device.is_compiled_with_cuda()); import paddleocr; print('PaddleOCR working!')"
```

**Expected output**:
```
CUDA: True
PaddleOCR working!
```

**Troubleshooting**: If CUDA shows False, check GPU drivers and CUDA installation

---

### 2. Check GPU Status
**Purpose**: Monitor GPU utilization and memory

**Command**:
```bash
nvidia-smi
```

**Information provided**:
- GPU temperature
- Memory usage
- Running processes
- Driver version

**When to use**:
- Before starting training
- Monitoring performance
- Troubleshooting GPU issues

---

### 3. Run Thai OCR Demo
**Purpose**: Test OCR functionality with Thai text images

**Command**:
```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='th', use_gpu=True); result = ocr.ocr('thai-letters/thai_ocr_dataset/images/000001.jpg'); print('Result:', result)"
```

**When to use**:
- Validating OCR accuracy
- Testing new images
- Debugging recognition issues

**Expected output**: Detected text with confidence scores

---

### 4. Test CRNN Model
**Purpose**: Verify CRNN model can be loaded successfully

**Command**:
```bash
cd thai-license-plate-recognition-CRNN && python -c "from keras.models import load_model; model = load_model('Model_LSTM+BN5--thai-v3.h5'); print('CRNN model loaded successfully!')"
```

**When to use**:
- After model training
- Before inference
- Validating model integrity

## Task Workflows 🔄

### Initial Setup Workflow
```
1. Install Thai OCR Dependencies
2. Setup Environment for RTX 5090
3. Test PaddleOCR Installation
4. Check GPU Status
```

### Dataset Preparation Workflow
```
1. Generate Thai Text Dataset
2. Create Thai OCR Dataset
3. Run Thai OCR Demo (to test)
```

### Training Workflow
```
1. Setup Environment for RTX 5090
2. Check GPU Status
3. Start CRNN Training
4. Test CRNN Model (after completion)
```

### Testing Workflow
```
1. Test PaddleOCR Installation
2. Check GPU Status
3. Run Thai OCR Demo
4. Test CRNN Model
```

## Performance Optimization Tips 🚀

### For RTX 5090 Users
1. Always run **"Setup Environment for RTX 5090"** before training
2. Monitor GPU memory with **"Check GPU Status"**
3. Use CUDA 12.6 for best compatibility
4. Set batch size according to available GPU memory

### Memory Management
- RTX 5090 has 24GB VRAM
- Recommended settings: 80% memory usage (19.2GB)
- Monitor temperature during training
- Use mixed precision for faster training

### Training Tips
- Start with small datasets for testing
- Use checkpoints to save progress
- Monitor validation accuracy
- Adjust learning rate based on convergence

## Troubleshooting Common Issues 🔧

### CUDA Not Available
**Symptoms**: CUDA: False in test output
**Solutions**:
1. Check NVIDIA driver version
2. Reinstall CUDA 12.6
3. Verify PATH environment variables
4. Run **"Check GPU Status"**

### Out of Memory Errors
**Symptoms**: CUDA out of memory during training
**Solutions**:
1. Run **"Setup Environment for RTX 5090"**
2. Reduce batch size
3. Use gradient accumulation
4. Clear GPU cache between runs

### Import Errors
**Symptoms**: ModuleNotFoundError for paddle/paddleocr
**Solutions**:
1. Run **"Install Thai OCR Dependencies"**
2. Check Python environment
3. Reinstall packages with **"Install PaddlePaddle GPU"**

### Model Loading Failures
**Symptoms**: Cannot load .h5 or .pdmodel files
**Solutions**:
1. Verify file paths
2. Check model compatibility
3. Run **"Test CRNN Model"** for diagnostics

## Custom Task Creation 📝

To add new tasks, edit `.vscode/tasks.json`:

```json
{
    "label": "Your Custom Task",
    "type": "shell",
    "command": "your-command-here",
    "group": "build", // or "test"
    "problemMatcher": [],
    "isBackground": false,
    "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
    }
}
```

## Integration with CI/CD 🔄

These tasks can be integrated into automated workflows:

### GitHub Actions Example
```yaml
- name: Install Dependencies
  run: cd thai-letters && pip install -r requirements.txt

- name: Test Installation
  run: python -c "import paddle; import paddleocr"

- name: Generate Dataset
  run: python thai-letters/create_thai_ocr_dataset.py
```

## Performance Benchmarks 📊

### Expected Performance (RTX 5090)
- **Installation**: 2-5 minutes
- **Dataset Generation**: 30 seconds - 2 minutes
- **OCR Inference**: 0.1-0.3 seconds per image
- **CRNN Training**: 2-5 seconds per batch
- **Model Loading**: 1-3 seconds

### Memory Usage
- **PaddleOCR**: 2-4GB GPU memory
- **CRNN Training**: 8-16GB GPU memory
- **Dataset Generation**: 1-2GB RAM

## Support and Documentation 📚

For additional help:
- [Installation Guide](../installation_guide.md)
- [Training Guide](../training_guide.md)
- [API Reference](../api_reference.md)
- [Troubleshooting](../troubleshooting.md)

## Contributing 🤝

When adding new tasks:
1. Follow naming convention: Action + Target (e.g., "Test CRNN Model")
2. Include proper error handling
3. Add documentation in this file
4. Test on RTX 5090 hardware
5. Consider cross-platform compatibility

---

*Last updated: July 21, 2025*
*Compatible with: RTX 5090, CUDA 12.6, PaddlePaddle 2.6+*
