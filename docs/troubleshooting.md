# Troubleshooting Guide for PaddleOCR Thai

## Common Issues and Solutions

### 1. Installation Issues

#### Problem: "no kernel image is available for execution on the device"
**Cause**: RTX 5090 (Compute Capability 12.0) not supported by pre-built PaddlePaddle

**Solutions**:
```bash
# Solution 1: Compile with SM 120 support
cmake .. -GNinja -DWITH_GPU=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="120"

# Solution 2: Use CUDA 12.6 instead of 12.8
# Download CUDA 12.6 from NVIDIA archives

# Solution 3: Fallback to CPU
pip uninstall paddlepaddle-gpu
pip install paddlepaddle
```

**Verification**:
```python
import paddle
print("CUDA compiled:", paddle.device.is_compiled_with_cuda())
print("GPU devices:", paddle.device.cuda.device_count())
```

#### Problem: CMake build fails on Windows
**Cause**: Missing Visual Studio components or environment variables

**Solutions**:
```batch
# Ensure Visual Studio 2019 with C++ workload is installed
# Use Native Tools Command Prompt
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

# Set CUDA environment
set CUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%PATH%

# Clean and retry
rmdir /s build
mkdir build && cd build
cmake .. -GNinja -DWITH_GPU=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="120"
```

#### Problem: CUDA version mismatch
**Cause**: Multiple CUDA versions or driver incompatibility

**Diagnosis**:
```batch
# Check CUDA toolkit version
nvcc --version

# Check driver version
nvidia-smi

# Check compatibility
python -c "import paddle; print(paddle.version.cuda())"
```

**Solutions**:
```batch
# Uninstall conflicting CUDA versions
# Keep only CUDA 12.6
# Update NVIDIA drivers to latest

# Verify installation
set CUDA_VISIBLE_DEVICES=0
python -c "import paddle; paddle.utils.run_check()"
```

### 2. Training Issues

#### Problem: Out of GPU memory during training
**Error**: `OutOfMemoryError: Out of memory error on GPU 0`

**Solutions**:
```yaml
# Reduce batch size in config
Train:
  loader:
    batch_size_per_card: 64  # Reduce from 128

# Enable gradient accumulation
Global:
  accumulate_grad_batch: 2

# Reduce image resolution
Train:
  dataset:
    transforms:
    - RecResizeImg:
        image_shape: [3, 48, 192]  # Reduce from [3, 64, 256]
```

**Memory optimization**:
```python
import paddle
import os

# Set memory fraction
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.7'

# Enable memory optimization
paddle.fluid.memory_optimize()
```

#### Problem: Training accuracy stuck at low values
**Cause**: Dataset issues, learning rate, or model configuration

**Diagnosis**:
```python
# Check dataset
python tools/train.py -c configs/rec/thai_svtr_tiny.yml --debug

# Visualize training data
import cv2
import matplotlib.pyplot as plt

def visualize_training_data():
    with open('thai-letters/thai_ocr_dataset/train_list.txt', 'r') as f:
        lines = f.readlines()[:10]
    
    for line in lines:
        img_path, label = line.strip().split('\t')
        img = cv2.imread(f'thai-letters/thai_ocr_dataset/{img_path}')
        plt.figure(figsize=(10, 2))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Label: {label}')
        plt.show()

visualize_training_data()
```

**Solutions**:
```yaml
# Adjust learning rate
Optimizer:
  lr:
    learning_rate: 0.0001  # Reduce learning rate
    warmup_epoch: 5

# Check character dictionary
Global:
  character_dict_path: ./ppocr/utils/dict/thai_dict.txt
  # Ensure all characters in labels exist in dictionary

# Increase training epochs
Global:
  epoch_num: 200  # Increase from 100
```

#### Problem: Model not learning Thai characters
**Cause**: Character dictionary mismatch or encoding issues

**Verification**:
```python
# Check character dictionary
with open('ppocr/utils/dict/thai_dict.txt', 'r', encoding='utf-8') as f:
    dict_chars = set(f.read().strip())

# Check labels
with open('thai-letters/thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
    label_chars = set()
    for line in f:
        if '\t' in line:
            label = line.strip().split('\t')[1]
            label_chars.update(set(label))

# Find missing characters
missing_chars = label_chars - dict_chars
print(f"Missing characters in dictionary: {missing_chars}")
```

**Fix**:
```python
# Update dictionary
def update_thai_dict():
    # Read existing dictionary
    with open('ppocr/utils/dict/thai_dict.txt', 'r', encoding='utf-8') as f:
        existing_chars = set(f.read().strip())
    
    # Read all characters from labels
    with open('thai-letters/thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
        all_chars = set()
        for line in f:
            if '\t' in line:
                label = line.strip().split('\t')[1]
                all_chars.update(set(label))
    
    # Combine and sort
    complete_chars = sorted(existing_chars | all_chars)
    
    # Write updated dictionary
    with open('ppocr/utils/dict/thai_dict.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(complete_chars))
    
    print(f"Updated dictionary with {len(complete_chars)} characters")

update_thai_dict()
```

### 3. Inference Issues

#### Problem: Low recognition accuracy on test images
**Cause**: Image quality, preprocessing, or model issues

**Diagnosis**:
```python
# Test image preprocessing
import cv2
import numpy as np
from paddleocr import PaddleOCR

def diagnose_image_quality(image_path):
    """Analyze image for OCR readiness"""
    img = cv2.imread(image_path)
    if img is None:
        return "Cannot read image"
    
    h, w = img.shape[:2]
    
    # Check dimensions
    if h < 32 or w < 32:
        return f"Image too small: {w}x{h}"
    
    # Check if grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Check contrast
    contrast = gray.std()
    if contrast < 20:
        return f"Low contrast: {contrast:.1f}"
    
    # Check brightness
    brightness = gray.mean()
    if brightness < 50 or brightness > 200:
        return f"Poor brightness: {brightness:.1f}"
    
    return "Image quality OK"

# Test image
result = diagnose_image_quality('test_image.jpg')
print(result)
```

**Solutions**:
```python
# Image enhancement
def enhance_image_for_ocr(image_path):
    """Enhance image for better OCR results"""
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Binarization
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# Use enhanced image
enhanced_img = enhance_image_for_ocr('test_image.jpg')
cv2.imwrite('enhanced_image.jpg', enhanced_img)

# Test OCR on enhanced image
ocr = PaddleOCR(rec_model_dir='./inference/thai_rec/')
result = ocr.ocr('enhanced_image.jpg', cls=True)
```

#### Problem: Text detection missing regions
**Cause**: Detection model threshold or image preprocessing

**Solutions**:
```python
# Adjust detection parameters
ocr = PaddleOCR(
    det_db_thresh=0.2,        # Lower threshold (default: 0.3)
    det_db_box_thresh=0.5,    # Lower box threshold (default: 0.6)
    det_db_unclip_ratio=2.0,  # Higher unclip ratio (default: 1.5)
    rec_model_dir='./inference/thai_rec/'
)
```

#### Problem: Wrong character recognition
**Cause**: Model not trained well or character dictionary issues

**Debug**:
```python
# Test individual character recognition
def test_character_recognition():
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(
        rec_model_dir='./inference/thai_rec/',
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt'
    )
    
    # Test on individual characters
    test_chars = ['ก', 'ข', 'ค', 'ง', '1', '2', '3']
    
    for char in test_chars:
        # Create simple image with character
        img = create_char_image(char)  # You need to implement this
        result = ocr.ocr(img)
        print(f"Expected: {char}, Got: {result}")

def create_char_image(char, size=(64, 64)):
    """Create simple image with single character"""
    import cv2
    import numpy as np
    
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, char, (10, 45), font, 1.5, (0, 0, 0), 2)
    return img
```

### 4. Performance Issues

#### Problem: Slow inference speed
**Cause**: CPU usage, large images, or inefficient preprocessing

**Optimization**:
```python
# GPU optimization
import os
os.environ['FLAGS_conv_workspace_size_limit'] = '512'
os.environ['FLAGS_max_inplace_grad_add'] = '5'

# Image resize before processing
def optimize_image_size(image_path, max_size=1920):
    """Resize image if too large"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite('resized_' + os.path.basename(image_path), img)
        return 'resized_' + os.path.basename(image_path)
    
    return image_path

# Batch processing
def batch_ocr_optimized(image_paths):
    """Optimized batch processing"""
    ocr = PaddleOCR(
        rec_model_dir='./inference/thai_rec/',
        use_gpu=True,
        show_log=False
    )
    
    results = []
    for img_path in image_paths:
        # Optimize image size
        optimized_path = optimize_image_size(img_path)
        
        # Process
        result = ocr.ocr(optimized_path, cls=True)
        results.append(result)
        
        # Cleanup
        if optimized_path != img_path:
            os.remove(optimized_path)
    
    return results
```

### 5. Model Integration Issues

#### Problem: Cannot load custom trained model
**Error**: `ValueError: Cannot load model from path`

**Solutions**:
```bash
# Check model files
ls -la inference/thai_rec/
# Should contain: inference.pdiparams, inference.pdmodel, inference.pdiparams.info

# Verify model export
python tools/export_model.py -c configs/rec/thai_svtr_tiny.yml \
    -o Global.pretrained_model=./output/rec_thai_svtr_tiny/best_accuracy \
       Global.save_inference_dir=./inference/thai_rec/

# Test model loading
python -c "
from paddleocr import PaddleOCR
try:
    ocr = PaddleOCR(rec_model_dir='./inference/thai_rec/')
    print('Model loaded successfully')
except Exception as e:
    print(f'Error: {e}')
"
```

#### Problem: Model conflicts with existing CRNN
**Cause**: Path conflicts or dependency issues

**Solution**:
```python
# Isolate environments
import sys
import os

class IsolatedOCR:
    """Isolated OCR to avoid conflicts"""
    
    def __init__(self, model_type='paddleocr'):
        self.model_type = model_type
        
        if model_type == 'paddleocr':
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                rec_model_dir='./inference/thai_rec/',
                rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt'
            )
        elif model_type == 'crnn':
            # Temporarily modify path for CRNN
            crnn_path = 'thai-license-plate-recognition-CRNN'
            if crnn_path not in sys.path:
                sys.path.insert(0, crnn_path)
            
            from keras.models import load_model
            self.ocr = load_model('thai-license-plate-recognition-CRNN/Model_LSTM+BN5--thai-v3.h5')
    
    def recognize(self, image_path):
        if self.model_type == 'paddleocr':
            return self.ocr.ocr(image_path, cls=True)
        else:
            # CRNN recognition logic
            pass

# Usage
paddle_ocr = IsolatedOCR('paddleocr')
crnn_ocr = IsolatedOCR('crnn')
```

### 6. Configuration Issues

#### Problem: YAML configuration not loading
**Error**: `yaml.scanner.ScannerError`

**Solutions**:
```yaml
# Check YAML syntax
# Common issues:
# 1. Tabs instead of spaces
# 2. Missing colons
# 3. Incorrect indentation

# Validate YAML
python -c "
import yaml
with open('configs/rec/thai_svtr_tiny.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
        print('YAML is valid')
    except yaml.YAMLError as e:
        print(f'YAML error: {e}')
"
```

#### Problem: Path issues in config
**Cause**: Relative paths not resolving correctly

**Fix**:
```yaml
# Use absolute paths in config
Global:
  character_dict_path: c:\Users\admin\Documents\paddlepadle\thai-letters\th_dict.txt
  
Train:
  dataset:
    data_dir: c:\Users\admin\Documents\paddlepadle\thai-letters\thai_ocr_dataset\
    label_file_list:
    - c:\Users\admin\Documents\paddlepadle\thai-letters\thai_ocr_dataset\train_list.txt
```

### 7. Data Issues

#### Problem: Training data format errors
**Error**: `IndexError: list index out of range`

**Diagnosis**:
```python
# Check data format
def validate_training_data():
    """Validate training data format"""
    with open('thai-letters/thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Line {i+1}: Invalid format - {line}")
                continue
            
            img_path, label = parts
            full_path = f'thai-letters/thai_ocr_dataset/images/{img_path}'
            
            if not os.path.exists(full_path):
                print(f"Line {i+1}: Missing image - {full_path}")
            
            if not label.strip():
                print(f"Line {i+1}: Empty label - {img_path}")

validate_training_data()
```

### 8. Environment Issues

#### Problem: Import errors
**Error**: `ModuleNotFoundError: No module named 'paddle'`

**Solutions**:
```bash
# Check installation
pip list | grep paddle

# Reinstall if necessary
pip uninstall paddlepaddle paddlepaddle-gpu paddleocr
pip install paddlepaddle-gpu
pip install paddleocr

# Check Python environment
python -c "import sys; print(sys.path)"
```

### 9. Logging and Debugging

#### Enable detailed logging
```python
import logging
import paddle

# Enable PaddlePaddle logging
paddle.set_printoptions(precision=4, sci_mode=False)

# Enable OCR logging
ocr = PaddleOCR(
    rec_model_dir='./inference/thai_rec/',
    show_log=True  # Enable logging
)

# Python logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Create debug script
```python
# debug_thai_ocr.py
import sys
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging

def comprehensive_debug():
    """Comprehensive debugging for Thai OCR"""
    
    print("=== PaddlePaddle Debug ===")
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
        print(f"GPU count: {paddle.device.cuda.device_count()}")
        if paddle.device.cuda.device_count() > 0:
            print(f"GPU name: {paddle.device.cuda.get_device_name()}")
    except Exception as e:
        print(f"PaddlePaddle error: {e}")
    
    print("\n=== OCR Model Debug ===")
    try:
        ocr = PaddleOCR(
            rec_model_dir='./inference/thai_rec/',
            rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
            use_gpu=True,
            show_log=True
        )
        print("OCR model loaded successfully")
    except Exception as e:
        print(f"OCR model error: {e}")
        return
    
    print("\n=== Test Image Debug ===")
    test_image = 'thai-letters/thai_ocr_dataset/images/000001.jpg'
    if os.path.exists(test_image):
        try:
            result = ocr.ocr(test_image, cls=True)
            print(f"OCR result: {result}")
        except Exception as e:
            print(f"OCR inference error: {e}")
    else:
        print(f"Test image not found: {test_image}")
    
    print("\n=== Dictionary Debug ===")
    dict_path = './ppocr/utils/dict/thai_dict.txt'
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read()
            print(f"Dictionary characters: {len(chars)}")
            print(f"Sample: {chars[:20]}...")
    else:
        print(f"Dictionary not found: {dict_path}")

if __name__ == "__main__":
    comprehensive_debug()
```

Run with: `python debug_thai_ocr.py`

## Getting Help

### Community Resources
- **PaddleOCR GitHub**: https://github.com/PaddlePaddle/PaddleOCR
- **PaddlePaddle Documentation**: https://paddlepaddle.org.cn/
- **Thai OCR Discussion**: Open issues on the project repository

### Reporting Issues
When reporting issues, include:
1. System information (OS, GPU, CUDA version)
2. PaddlePaddle and PaddleOCR versions
3. Full error messages and stack traces
4. Sample images that reproduce the issue
5. Configuration files used

### Performance Baselines
Expected performance on RTX 5090:
- **Training speed**: 2-5 seconds per batch (128 images)
- **Inference speed**: 0.1-0.3 seconds per image
- **Memory usage**: 4-8GB GPU memory during training
- **Accuracy**: 85-95% on clear Thai text
