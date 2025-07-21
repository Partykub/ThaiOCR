# Amazon Q Rules for PaddleOCR Thai Project

## Project Overview

This project provides Thai language support for PaddleOCR, including custom models, training datasets, and configuration files specifically designed for Thai text recognition. The project integrates with existing CRNN models and supports RTX 5090 GPU with CUDA 12.6.

## Project Structure and Context

```
paddlepadle/
├── thai-letters/                    # Thai OCR dataset and utilities
│   ├── thai_ocr_dataset/           # Training images and labels (88+ images)
│   │   ├── images/                 # 000000.jpg to 000087.jpg
│   │   └── labels.txt              # Image-text label pairs
│   ├── thai_text_generator.py      # Text generation utilities
│   ├── th_dict.txt                 # Thai character dictionary
│   ├── thai_corpus.txt             # Thai text corpus
│   └── requirements.txt            # Python dependencies
├── thai-license-plate-recognition-CRNN/  # Existing CRNN model (Keras/TensorFlow)
│   ├── Model_LSTM+BN5--thai-v3.h5  # Trained CRNN model
│   ├── Model.py                    # CRNN model architecture
│   ├── parameter.py                # Model parameters
│   └── training.py                 # Training script
├── build-model-th/                 # Model building utilities
└── docs/                           # Comprehensive documentation
    ├── README.md                   # Project overview
    ├── installation_guide.md       # RTX 5090 setup guide
    ├── training_guide.md           # Model training guide
    ├── api_reference.md            # API documentation
    ├── troubleshooting.md          # Common issues and solutions
    └── requirements.md             # Dependencies list
```

## Key Technical Details

### Hardware and Environment
- **Primary GPU**: RTX 5090 with Compute Capability 12.0 (SM 120)
- **CUDA Version**: 12.6 (recommended for RTX 5090 compatibility)
- **cuDNN**: 9.0
- **Python**: 3.10-3.11 (64-bit)
- **OS**: Windows 11 (primary), Windows 10, Linux

### Core Dependencies
```python
# Essential packages
paddlepaddle-gpu>=2.6.0  # GPU version for RTX 5090
paddleocr>=2.7.0
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.21.0

# For CRNN compatibility
tensorflow>=2.3.0,<2.16.0
keras>=2.3.0
h5py>=3.1.0
```

### Critical Configuration for RTX 5090
```bash
# CMake configuration for PaddlePaddle compilation
cmake .. -GNinja \
  -DWITH_GPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_UNITY_BUILD=ON \
  -DCUDA_ARCH_NAME=Manual \
  -DCUDA_ARCH_BIN="120"  # Critical for RTX 5090 support
```

## Code Patterns and Best Practices

### 1. Thai OCR Initialization
```python
from paddleocr import PaddleOCR

# Standard Thai OCR setup
ocr = PaddleOCR(
    use_angle_cls=True,
    rec_model_dir='./inference/thai_rec/',
    rec_char_dict_path='./thai-letters/th_dict.txt',
    lang='th',
    use_gpu=True,
    show_log=False
)
```

### 2. Training Configuration Template
```yaml
Global:
  use_gpu: true
  character_dict_path: ./thai-letters/th_dict.txt
  character_type: thai
  max_text_length: 25

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  
Train:
  dataset:
    data_dir: ./thai-letters/thai_ocr_dataset/
    label_file_list:
    - ./thai-letters/thai_ocr_dataset/train_list.txt
```

### 3. Hybrid Model Integration Pattern
```python
class HybridThaiOCR:
    """Combines PaddleOCR with existing CRNN model"""
    def __init__(self):
        # PaddleOCR for general text
        self.paddle_ocr = PaddleOCR(
            rec_model_dir='./inference/thai_rec/',
            rec_char_dict_path='./thai-letters/th_dict.txt'
        )
        
        # CRNN for license plates
        try:
            from keras.models import load_model
            self.crnn_model = load_model(
                'thai-license-plate-recognition-CRNN/Model_LSTM+BN5--thai-v3.h5'
            )
        except:
            self.crnn_model = None
```

### 4. Error Handling Pattern for RTX 5090
```python
def safe_gpu_initialization():
    """Handle RTX 5090 specific issues"""
    try:
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("CUDA not available")
        
        # Check compute capability
        if paddle.device.cuda.device_count() > 0:
            # RTX 5090 specific checks
            ocr = PaddleOCR(use_gpu=True)
            return ocr
    except Exception as e:
        # Fallback to CPU
        print(f"GPU initialization failed: {e}")
        return PaddleOCR(use_gpu=False)
```

## Common Issues and Solutions

### RTX 5090 Compatibility
- **Issue**: "no kernel image is available" error
- **Solution**: Compile PaddlePaddle with `-DCUDA_ARCH_BIN="120"`
- **Fallback**: Use CPU version or CUDA 12.6 instead of 12.8

### Memory Optimization
```python
# GPU memory settings for RTX 5090
import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'
os.environ['FLAGS_conv_workspace_size_limit'] = '512'
```

### Training Data Format
```
# labels.txt format
000001.jpg	ภาษาไทย
000002.jpg	ตัวอย่างข้อความ
000003.jpg	123456
```

## File Naming Conventions

### Dataset Files
- Images: `000000.jpg` to `000087.jpg` (zero-padded)
- Labels: `labels.txt` (tab-separated: filename\tlabel)
- Dictionary: `th_dict.txt` (Thai characters)

### Model Files
- PaddleOCR: `./inference/thai_rec/` (inference.pdmodel, inference.pdiparams)
- CRNN: `Model_LSTM+BN5--thai-v3.h5`

### Configuration Files
- Training config: `thai_svtr_tiny.yml`
- Requirements: `requirements.txt`

## Performance Expectations

### RTX 5090 Performance
- **Inference speed**: 0.1-0.3 seconds per image
- **Memory usage**: 4-8GB GPU memory during training
- **Training speed**: 2-5 seconds per batch (128 images)
- **Accuracy target**: 85-95% on clear Thai text

### Model Comparison
| Feature | PaddleOCR Thai | CRNN (Existing) |
|---------|----------------|-----------------|
| Framework | PaddlePaddle | Keras/TensorFlow |
| Detection | Built-in | Manual preprocessing |
| Recognition | Thai + Multilingual | Thai license plates only |
| Deployment | Production ready | Prototype |

## Testing Patterns

### Unit Test Example
```python
def test_thai_ocr():
    """Test Thai OCR with sample images"""
    ocr = PaddleOCR(rec_model_dir='./inference/thai_rec/')
    
    test_images = [
        'thai-letters/thai_ocr_dataset/images/000001.jpg',
        'thai-letters/thai_ocr_dataset/images/000002.jpg'
    ]
    
    for img_path in test_images:
        result = ocr.ocr(img_path, cls=True)
        assert result is not None
        assert len(result) > 0
```

### Benchmark Test
```python
def benchmark_models():
    """Compare PaddleOCR vs CRNN performance"""
    import time
    
    # Test both models on same dataset
    # Measure accuracy, speed, memory usage
    # Generate comparison report
```

## Environment Variables

```bash
# CUDA settings for RTX 5090
set CUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%PATH%

# PaddlePaddle optimization
set FLAGS_fraction_of_gpu_memory_to_use=0.8
set FLAGS_conv_workspace_size_limit=512
set FLAGS_cudnn_deterministic=true
```

## Integration Guidelines

### Web API Pattern
```python
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)
ocr = PaddleOCR(rec_model_dir='./inference/thai_rec/')

@app.route('/ocr', methods=['POST'])
def recognize_text():
    # Decode base64 image
    # Run OCR
    # Return JSON results
```

### Batch Processing Pattern
```python
def batch_process_images(image_folder):
    """Process multiple images efficiently"""
    ocr = PaddleOCR(use_gpu=True, show_log=False)
    
    results = {}
    for img_file in glob.glob(f"{image_folder}/*.jpg"):
        result = ocr.ocr(img_file, cls=True)
        results[img_file] = format_result(result)
    
    return results
```

## Documentation References

When generating code for this project, refer to:
- **Installation**: `/docs/installation_guide.md` for setup procedures
- **Training**: `/docs/training_guide.md` for model training
- **API Usage**: `/docs/api_reference.md` for integration examples
- **Troubleshooting**: `/docs/troubleshooting.md` for common issues
- **Requirements**: `/docs/requirements.md` for dependencies

## Code Generation Guidelines

1. **Always check RTX 5090 compatibility** when generating GPU-related code
2. **Use absolute paths** for Windows file operations
3. **Include error handling** for CUDA availability
4. **Prefer PaddleOCR** for new features, CRNN for license plate specific tasks
5. **Follow Thai character encoding** (UTF-8) for text processing
6. **Include performance monitoring** in training and inference code
7. **Add logging** for debugging RTX 5090 specific issues

## Model Integration Priority

1. **PaddleOCR Thai model** - Primary choice for general Thai text
2. **CRNN model** - Fallback for license plate recognition
3. **Hybrid approach** - Combine both based on use case
4. **CPU fallback** - When GPU issues occur

This project bridges modern PaddleOCR capabilities with existing CRNN infrastructure while supporting cutting-edge RTX 5090 hardware.