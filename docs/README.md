# PaddleOCR Thai Language Support

## Overview
This project provides Thai language support for PaddleOCR, including custom models, training datasets, and configuration files specifically designed for Thai text recognition.

## Project Structure
```
paddlepadle/
├── thai-letters/                    # Thai OCR dataset and utilities
│   ├── thai_ocr_dataset/           # Training images and labels
│   ├── thai_text_generator.py      # Text generation utilities
│   ├── th_dict.txt                 # Thai character dictionary
│   └── requirements.txt            # Python dependencies
├── thai-license-plate-recognition-CRNN/  # Existing CRNN model
├── build-model-th/                 # Model building utilities
└── docs/                           # Documentation
```

## Quick Start

### 1. Installation
```bash
# Install CUDA 12.6 and cuDNN 9.0 first
pip install -r thai-letters/requirements.txt
```

### 2. Basic Usage
```python
from paddleocr import PaddleOCR
import cv2

# Initialize Thai OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='th',  # Thai language
    rec_char_dict_path='./thai-letters/th_dict.txt'
)

# Process image
result = ocr.ocr('path/to/thai_image.jpg', cls=True)
print(result)
```

### 3. Training Custom Model
See [Training Guide](training_guide.md) for detailed instructions.

## Features
- ✅ Thai character recognition
- ✅ License plate recognition (CRNN model)
- ✅ Text detection and recognition pipeline
- ✅ Custom dataset generation
- ✅ RTX 5090 compatibility (CUDA 12.6)

## Hardware Requirements
- **GPU**: RTX 5090 or compatible NVIDIA GPU
- **CUDA**: 12.6 (recommended for RTX 5090)
- **cuDNN**: 9.0
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and datasets

## Documentation
- [Installation Guide](installation_guide.md)
- [Training Guide](training_guide.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)

## Comparison with Existing CRNN Model
| Feature | PaddleOCR Thai | CRNN (Existing) |
|---------|----------------|-----------------|
| Framework | PaddlePaddle | Keras/TensorFlow |
| Detection | Built-in | Manual preprocessing |
| Recognition | Thai + Multilingual | Thai license plates only |
| Deployment | Production ready | Prototype |
| Community | Active | Limited |

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License.

## Contact
For questions and support, please open an issue on GitHub.
