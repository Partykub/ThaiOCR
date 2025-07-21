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

**Commands**:
```bash
# สร้าง Dataset ตัวอย่าง (500 ภาพ)
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 500 -o thai-letters/generated_text_samples

# สร้าง Dataset ครบชุด (9,672 ภาพ - เท่ากับจำนวนบรรทัดใน corpus)
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 9672 -o thai-letters/full_corpus_dataset

# รันสคริปต์อัตโนมัติ
python thai-letters/create_all_datasets.py

# หรือใช้ batch file (Windows)
thai-letters/create_datasets.bat
```

**Parameters**:
- `-c, --corpus`: ไฟล์ corpus (default: thai_corpus.txt)
- `-n, --num`: จำนวนภาพที่ต้องการ (default: 1000)  
- `-o, --output`: โฟลเดอร์ output (default: ./dataset/)

**When to use**:
- Need more training data
- Expanding vocabulary coverage
- Creating specialized text samples

**Output**: Image datasets with corresponding labels

**Result**: 
- Dataset ตัวอย่าง: 500 ภาพ
- Dataset ครบชุด: 9,672 ภาพ  
- รวมกับเดิม: 15,172 ภาพทั้งหมด

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
**Purpose**: Begin training the CRNN model for license plate recognition with RTX 5090 GPU

**🚨 MANDATORY GPU REQUIREMENTS**:
- RTX 5090 GPU REQUIRED - NO CPU TRAINING ALLOWED
- GPU verification MUST pass before training begins
- Training automatically ABORTS if no GPU detected

**Commands**:
```bash
# STEP 1: Enforce GPU Training (MANDATORY)
cd build-model-th && python enforce_gpu_training.py

# STEP 2: Start CRNN Training (RTX 5090 Optimized)
cd build-model-th && python start_crnn_training.py

# Alternative (Windows Batch)
build-model-th\start_crnn_training.bat
```

**VS Code Tasks**:
1. **Ctrl+Shift+P** → **"Tasks: Run Task"** → **"Enforce GPU Training (MANDATORY)"**
2. **Ctrl+Shift+P** → **"Tasks: Run Task"** → **"Start CRNN Training"**

**When to use**:
- Training new CRNN models for Thai license plates
- Fine-tuning existing models with RTX 5090 performance
- Experimenting with training parameters
- Production model training

**🎮 RTX 5090 Features**:
- ⚡ **XLA JIT Compilation**: Enabled for maximum performance
- 🔄 **Mixed Precision**: Float16 training for RTX 5090
- 💾 **Dynamic Memory**: 80% GPU memory allocation (19.2GB)
- 🎯 **Optimizer Settings**: All TensorFlow optimizations enabled
- 📊 **Automatic Monitoring**: GPU status and performance tracking

**⚠️ Error Handling**:
- `CRITICAL: NO GPU DETECTED - TRAINING ABORTED` 
- `CRITICAL: CUDA not available - TRAINING CANNOT PROCEED`
- `CRITICAL: GPU NOT AVAILABLE - TRAINING CANNOT PROCEED`

**📊 Expected Performance**:
- Training Duration: 1-2 minutes (with early stopping)
- Epochs: 15-50 (automatic early stopping)
- GPU Memory Usage: 8-16GB
- Model Size: ~85MB (7.4M parameters)
- Batch Size: 64 (RTX 5090 optimized)

**Note**: 
- 🎮 **GPU MANDATORY**: Training will FAIL if no GPU available
- ⚡ **Background Task**: May run for 1-2 hours for full training
- 📊 **Monitoring**: Use monitoring tools to track progress
- 🔄 **Checkpoints**: Automatic model saving every epoch

---

### 6. Setup Environment for RTX 5090
**Purpose**: Configure optimal environment variables and settings for RTX 5090 GPU performance

**🎮 RTX 5090 Features**:
- ⚡ **24GB VRAM**: Complete RTX 5090 Laptop GPU detected and configured
- 🔥 **21,760 CUDA Cores**: Maximum parallel processing capability
- 💾 **Memory Management**: 80% allocation (19.2GB usable)
- 🚀 **Mixed Precision**: TF32/FP16 training optimization
- 🎯 **XLA Compilation**: GPU JIT compilation enabled

**Commands**:
```bash
# STEP 1: Full RTX 5090 Environment Setup (Recommended)
python build-model-th/setup_rtx5090_environment.py

# STEP 2: Apply Environment (Windows)
build-model-th\setup_rtx5090_env.bat

# Quick Environment Setup (Basic)
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
```

**VS Code Tasks**:
1. **Ctrl+Shift+P** → **"Tasks: Run Task"** → **"Setup RTX 5090 Environment"**
2. **Ctrl+Shift+P** → **"Tasks: Run Task"** → **"Apply RTX 5090 Environment (Windows)"**

**When to use**:
- ✅ **MANDATORY**: Before ANY training operations
- 🔧 Before training large models
- 💾 When facing GPU memory issues
- 🚀 Optimizing performance for RTX 5090
- 🎮 Initial RTX 5090 setup and configuration

**🔧 Environment Variables Configured**:
- **GPU Memory**: `FLAGS_fraction_of_gpu_memory_to_use=0.8` (19.2GB)
- **Workspace**: `FLAGS_conv_workspace_size_limit=512` (512MB)
- **Deterministic**: `FLAGS_cudnn_deterministic=true` (reproducible)
- **TensorFlow**: GPU allocator, XLA, mixed precision
- **CUDA**: Device ordering, caching, profiling
- **Performance**: Thread optimization, memory reuse

**📊 Expected Results**:
```
✅ RTX 5090 DETECTED: NVIDIA GeForce RTX 5090 Laptop GPU (24GB)
✅ PaddlePaddle GPU: 1 device(s)
✅ Environment variables: 20+ optimizations configured
✅ Performance files created: .env.rtx5090, batch scripts, reports
```

**📄 Files Created**:
- `.env.rtx5090` - Environment variables file
- `build-model-th/setup_rtx5090_env.bat` - Windows setup script
- `build-model-th/rtx5090_setup_report.md` - Performance report
- `.vscode/tasks.json` - VS Code task integration

**⚡ Performance Impact**:
- **Training Speed**: 15-20x faster than CPU
- **Memory Efficiency**: 80% GPU utilization (19.2GB)
- **Batch Processing**: 64 samples per batch (optimized)
- **Convergence**: Faster training with mixed precision

**Effect**: Configures RTX 5090 for maximum performance, memory optimization, and training acceleration

## Test Tasks 🧪

### 0. Enforce GPU Training (MANDATORY)
**Purpose**: Verify RTX 5090 GPU availability and enforce GPU-only training policy

**Command**:
```bash
cd build-model-th && python enforce_gpu_training.py
```

**VS Code Task**: **"Enforce GPU Training (MANDATORY)"**

**Expected output**:
```
✅ GPU DETECTED: 1 GPU(s) found
✅ RTX 5090 DETECTED: /physical_device:GPU:0
✅ GPU COMPUTATION TEST PASSED
🎮 GPU VERIFICATION SUCCESSFUL - TRAINING AUTHORIZED
```

**When to use**:
- **MANDATORY**: Before ANY training operation
- Verifying RTX 5090 configuration
- Troubleshooting GPU issues
- Setting up RTX 5090 optimizations

**Features**:
- 🎮 **RTX 5090 Detection**: Verifies specific GPU model
- ⚡ **Performance Setup**: Configures optimal settings
- 🔍 **CUDA Testing**: Tests actual GPU computation
- 🚫 **Training Block**: Prevents CPU training attempts

---

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
1. Enforce GPU Training (MANDATORY) - MUST PASS
2. Setup Environment for RTX 5090
3. Validate CRNN Training Data
4. Start CRNN Training (GPU Enforced)
5. Monitor Training Progress
6. Test CRNN Model (after completion)
```

### Testing Workflow
```
1. Enforce GPU Training (MANDATORY)
2. Test PaddleOCR Installation  
3. Check GPU Status
4. Run Thai OCR Demo
5. Test CRNN Model
```

## Performance Optimization Tips 🚀

### For RTX 5090 Users
1. **MANDATORY**: Run **"Enforce GPU Training (MANDATORY)"** before ANY training
2. Always run **"Setup Environment for RTX 5090"** before training
3. Monitor GPU memory with **"Check GPU Status"**
4. Use CUDA 12.6 for best compatibility
5. Set batch size according to available GPU memory

### GPU Training Requirements
- **RTX 5090**: 24GB VRAM available
- **CUDA 12.6**: Mandatory for RTX 5090 support
- **TensorFlow-GPU**: Required (no CPU fallback)
- **Batch Size**: 64 (optimized for RTX 5090)
- **Memory Usage**: 80% allocation (19.2GB)

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

### GPU Training Failures (NEW)
**Symptoms**: "CRITICAL: NO GPU DETECTED - TRAINING ABORTED"
**Solutions**:
1. Run **"Enforce GPU Training (MANDATORY)"** first
2. Check RTX 5090 GPU is connected and powered
3. Verify NVIDIA drivers are up to date
4. Ensure CUDA 12.6 is properly installed
5. Check Windows Device Manager for GPU status
6. Run `nvidia-smi` to verify GPU detection

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
- **GPU Enforcement**: 2-5 seconds
- **Installation**: 2-5 minutes
- **Dataset Generation**: 30 seconds - 2 minutes
- **OCR Inference**: 0.1-0.3 seconds per image
- **CRNN Training**: 2-5 seconds per batch (GPU) vs 30-60 seconds (CPU)
- **Model Loading**: 1-3 seconds

### GPU Training Performance
- **Training Speed**: 15-20x faster than CPU
- **Memory Efficiency**: 80% GPU memory utilization
- **Batch Processing**: 64 samples per batch (RTX 5090)
- **Convergence**: Early stopping around epoch 15
- **Total Training Time**: 1.4 minutes (vs 20-30 minutes CPU)

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
