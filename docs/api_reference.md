# API Reference for PaddleOCR Thai

## Overview
Complete API reference for using PaddleOCR with Thai language support, including integration examples and best practices.

## Core Classes

### ThaiPaddleOCR

Main class for Thai text recognition.

```python
class ThaiPaddleOCR:
    def __init__(
        self,
        use_gpu=True,
        det_model_dir=None,
        rec_model_dir=None,
        cls_model_dir=None,
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
        use_angle_cls=True,
        show_log=False
    ):
        """
        Initialize Thai OCR engine
        
        Args:
            use_gpu (bool): Use GPU acceleration
            det_model_dir (str): Path to detection model
            rec_model_dir (str): Path to recognition model  
            cls_model_dir (str): Path to classification model
            rec_char_dict_path (str): Path to Thai character dictionary
            use_angle_cls (bool): Use text angle classification
            show_log (bool): Show debug logs
        """
```

#### Methods

##### ocr(img_path, cls=True)
Perform complete OCR on image.

```python
def ocr(self, img_path, cls=True):
    """
    Recognize text in image
    
    Args:
        img_path (str): Path to image file
        cls (bool): Use angle classification
        
    Returns:
        list: Detection and recognition results
        
    Example:
        >>> ocr = ThaiPaddleOCR()
        >>> result = ocr.ocr('thai_text.jpg')
        >>> print(result)
        [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('recognized_text', confidence)]]
    """
```

##### detect_text(img_path)
Text detection only.

```python
def detect_text(self, img_path):
    """
    Detect text regions in image
    
    Args:
        img_path (str): Path to image file
        
    Returns:
        list: Bounding boxes of detected text regions
        
    Example:
        >>> boxes = ocr.detect_text('image.jpg')
        >>> for box in boxes:
        ...     print(f"Text region: {box}")
    """
```

##### recognize_text(img_crop)
Text recognition only.

```python
def recognize_text(self, img_crop):
    """
    Recognize text in cropped image
    
    Args:
        img_crop (numpy.ndarray): Cropped text image
        
    Returns:
        tuple: (recognized_text, confidence_score)
        
    Example:
        >>> import cv2
        >>> img = cv2.imread('cropped_text.jpg')
        >>> text, confidence = ocr.recognize_text(img)
        >>> print(f"Text: {text}, Confidence: {confidence}")
    """
```

## Integration Examples

### 1. Basic Usage

```python
from paddleocr import PaddleOCR
import cv2

# Initialize OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    rec_model_dir='./inference/thai_rec/',
    rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
    lang='th',
    use_gpu=True
)

# Process single image
result = ocr.ocr('thai_document.jpg', cls=True)

# Print results
for idx, line in enumerate(result):
    for text_box in line:
        points, (text, confidence) = text_box
        print(f"Text {idx}: {text} (confidence: {confidence:.3f})")
        print(f"Coordinates: {points}")
```

### 2. Batch Processing

```python
import os
import glob
from paddleocr import PaddleOCR
import json

def batch_ocr_processing(image_folder, output_file):
    """Process multiple images and save results"""
    
    ocr = PaddleOCR(
        rec_model_dir='./inference/thai_rec/',
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
        use_gpu=True,
        show_log=False
    )
    
    results = {}
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    
    for img_path in image_files:
        print(f"Processing: {img_path}")
        try:
            result = ocr.ocr(img_path, cls=True)
            results[img_path] = {
                'status': 'success',
                'text_regions': []
            }
            
            for line in result:
                for text_box in line:
                    points, (text, confidence) = text_box
                    results[img_path]['text_regions'].append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': points
                    })
                    
        except Exception as e:
            results[img_path] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

# Usage
results = batch_ocr_processing('./test_images/', 'ocr_results.json')
```

### 3. Integration with Existing CRNN Model

```python
import sys
sys.path.append('thai-license-plate-recognition-CRNN')

from paddleocr import PaddleOCR
from keras.models import load_model
import cv2
import numpy as np

class HybridThaiOCR:
    """Combines PaddleOCR with existing CRNN model"""
    
    def __init__(self):
        # PaddleOCR for general text
        self.paddle_ocr = PaddleOCR(
            rec_model_dir='./inference/thai_rec/',
            rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
            use_gpu=True
        )
        
        # CRNN for license plates
        try:
            self.crnn_model = load_model(
                'thai-license-plate-recognition-CRNN/Model_LSTM+BN5--thai-v3.h5'
            )
            self.use_crnn = True
        except:
            print("CRNN model not available, using PaddleOCR only")
            self.use_crnn = False
    
    def recognize(self, image_path, is_license_plate=False):
        """
        Recognize text using appropriate model
        
        Args:
            image_path (str): Path to image
            is_license_plate (bool): Use CRNN for license plates
            
        Returns:
            dict: Recognition results
        """
        if is_license_plate and self.use_crnn:
            return self._recognize_license_plate(image_path)
        else:
            return self._recognize_general(image_path)
    
    def _recognize_general(self, image_path):
        """General text recognition with PaddleOCR"""
        result = self.paddle_ocr.ocr(image_path, cls=True)
        
        extracted_text = []
        for line in result:
            for text_box in line:
                points, (text, confidence) = text_box
                extracted_text.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': points,
                    'model': 'paddleocr'
                })
        
        return {
            'success': True,
            'texts': extracted_text,
            'model_used': 'paddleocr'
        }
    
    def _recognize_license_plate(self, image_path):
        """License plate recognition with CRNN"""
        # Implementation for CRNN model
        # This would use the existing CRNN pipeline
        return {
            'success': True,
            'texts': [],  # CRNN results would go here
            'model_used': 'crnn'
        }

# Usage
hybrid_ocr = HybridThaiOCR()

# General text
result1 = hybrid_ocr.recognize('document.jpg', is_license_plate=False)

# License plate
result2 = hybrid_ocr.recognize('license_plate.jpg', is_license_plate=True)
```

### 4. Web API Integration

```python
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Initialize OCR once
ocr = PaddleOCR(
    rec_model_dir='./inference/thai_rec/',
    rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
    use_gpu=True,
    show_log=False
)

@app.route('/ocr', methods=['POST'])
def recognize_text():
    """
    OCR API endpoint
    
    Request:
        {
            "image": "base64_encoded_image",
            "use_angle_cls": true
        }
    
    Response:
        {
            "success": true,
            "results": [
                {
                    "text": "recognized_text",
                    "confidence": 0.95,
                    "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                }
            ],
            "processing_time": 0.123
        }
    """
    try:
        import time
        start_time = time.time()
        
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert PIL to OpenCV format
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        # Run OCR
        use_cls = data.get('use_angle_cls', True)
        result = ocr.ocr(image_cv, cls=use_cls)
        
        # Format results
        formatted_results = []
        for line in result:
            for text_box in line:
                points, (text, confidence) = text_box
                formatted_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': points
                })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'processing_time': processing_time
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'thai_paddleocr'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## Configuration Options

### Model Configuration

```python
# Detection model configuration
det_config = {
    'det_algorithm': 'DB',
    'det_model_dir': './inference/thai_det/',
    'det_limit_side_len': 960,
    'det_limit_type': 'max',
    'det_db_thresh': 0.3,
    'det_db_box_thresh': 0.6,
    'det_db_unclip_ratio': 1.5
}

# Recognition model configuration  
rec_config = {
    'rec_algorithm': 'SVTR_LCNet',
    'rec_model_dir': './inference/thai_rec/',
    'rec_char_dict_path': './ppocr/utils/dict/thai_dict.txt',
    'rec_image_shape': '3, 64, 256',
    'rec_batch_num': 6
}

# Initialize with custom config
ocr = PaddleOCR(**det_config, **rec_config)
```

### Performance Tuning

```python
# GPU memory optimization
import paddle
paddle.device.set_device('gpu:0')

# Set memory fraction
import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'

# Enable memory optimization
os.environ['FLAGS_conv_workspace_size_limit'] = '512'
```

## Error Handling

### Common Error Patterns

```python
from paddleocr import PaddleOCR
import cv2
import logging

def robust_ocr(image_path, max_retries=3):
    """
    Robust OCR with error handling and retries
    """
    ocr = PaddleOCR(
        rec_model_dir='./inference/thai_rec/',
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
        use_gpu=True,
        show_log=False
    )
    
    for attempt in range(max_retries):
        try:
            # Validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Check if image is readable
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Check image dimensions
            h, w = img.shape[:2]
            if h < 32 or w < 32:
                raise ValueError(f"Image too small: {w}x{h}")
            
            # Run OCR
            result = ocr.ocr(image_path, cls=True)
            
            if result is None or len(result) == 0:
                raise RuntimeError("OCR returned no results")
            
            return {
                'success': True,
                'result': result,
                'attempts': attempt + 1
            }
            
        except Exception as e:
            logging.warning(f"OCR attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return {
                    'success': False,
                    'error': str(e),
                    'attempts': max_retries
                }
    
    return {'success': False, 'error': 'Max retries exceeded'}

# Usage
result = robust_ocr('test_image.jpg')
if result['success']:
    print("OCR successful:", result['result'])
else:
    print("OCR failed:", result['error'])
```

## Best Practices

### 1. Image Preprocessing

```python
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image for better OCR results
    """
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Resize if too small
    h, w = denoised.shape
    if h < 64 or w < 256:
        scale = max(64/h, 256/w)
        new_h, new_w = int(h*scale), int(w*scale)
        denoised = cv2.resize(denoised, (new_w, new_h))
    
    return denoised
```

### 2. Confidence Filtering

```python
def filter_by_confidence(ocr_result, min_confidence=0.7):
    """
    Filter OCR results by confidence threshold
    """
    filtered_results = []
    
    for line in ocr_result:
        for text_box in line:
            points, (text, confidence) = text_box
            if confidence >= min_confidence:
                filtered_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': points
                })
    
    return filtered_results
```

### 3. Text Post-processing

```python
import re

def postprocess_thai_text(text):
    """
    Clean and normalize Thai text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common OCR errors for Thai
    # This would include Thai-specific corrections
    corrections = {
        'ๅ': 'ฯ',  # Example correction
        # Add more based on your data
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text
```

## Performance Metrics

### Benchmark Results

```python
def benchmark_thai_ocr():
    """
    Benchmark Thai OCR performance
    """
    import time
    import glob
    
    ocr = PaddleOCR(
        rec_model_dir='./inference/thai_rec/',
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
        use_gpu=True
    )
    
    test_images = glob.glob('./test_images/*.jpg')
    total_time = 0
    successful_ocr = 0
    
    for img_path in test_images:
        start_time = time.time()
        try:
            result = ocr.ocr(img_path, cls=True)
            if result and len(result) > 0:
                successful_ocr += 1
        except:
            pass
        total_time += time.time() - start_time
    
    avg_time = total_time / len(test_images)
    success_rate = successful_ocr / len(test_images)
    
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Images per second: {1/avg_time:.1f}")

# Run benchmark
benchmark_thai_ocr()
```

Expected performance on RTX 5090:
- **Processing speed**: 0.1-0.3 seconds per image
- **Memory usage**: 2-4GB GPU memory
- **Accuracy**: 85-95% on clear Thai text

## Next Steps

- [Deployment Guide](deployment_guide.md) - Deploy to production
- [Troubleshooting](troubleshooting.md) - Solve common issues
- [Examples](examples/) - More integration examples
