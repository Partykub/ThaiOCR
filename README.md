# Thai OCR - CRNN Model with CTC Loss

A clean, optimized Thai Optical Character Recognition system using CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) loss, specifically optimized for RTX 5090 GPU.

## 📁 Project Structure

```
├── src/                     # Source code
│   ├── models/              # Model architecture
│   │   └── thai_crnn.py     # CRNN model definition
│   ├── training/            # Training scripts
│   │   └── train_thai_crnn_clean.py
│   ├── testing/             # Testing scripts
│   │   └── test_thai_crnn_clean.py
│   └── utils/               # Utilities
│       ├── dataset.py       # Dataset classes
│       └── gpu_utils.py     # GPU utilities
├── scripts/                 # Batch scripts
│   ├── train_model.bat      # Training script
│   └── test_model.bat       # Testing script
├── models/                  # Trained models
│   ├── thai_crnn_ctc_best.pth
│   └── thai_char_map.json
├── configs/                 # Configuration files
│   └── model_config.json
├── logs/                    # Training logs and results
├── thai-letters/            # Dataset
│   └── thai_ocr_dataset/
└── docs/                    # Documentation
```

## 🚀 Quick Start

### Prerequisites
- RTX 5090 GPU (or compatible CUDA GPU)
- PyTorch 2.9+ with CUDA support
- Python 3.8+

### Training
```bash
# Option 1: Use batch script (Windows)
scripts\train_model.bat

# Option 2: Direct Python command
tf_gpu_env\Scripts\python.exe src\training\train_thai_crnn_clean.py
```

### Testing
```bash
# Option 1: Use batch script (Windows)
scripts\test_model.bat

# Option 2: Direct Python command
tf_gpu_env\Scripts\python.exe src\testing\test_thai_crnn_clean.py
```

## 🏗️ Model Architecture

- **CNN Feature Extractor**: 4 convolutional blocks with batch normalization
- **RNN Sequence Modeling**: 2-layer bidirectional LSTM
- **CTC Output**: Connectionist Temporal Classification for variable-length sequences
- **Input Size**: 64×128 RGB images
- **Output**: Variable-length Thai text sequences

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Loss Function | CTC Loss |

## 🎯 Features

- **RTX 5090 Optimized**: Specific memory allocation and compatibility settings
- **Clean Architecture**: Modular design with separated concerns
- **CTC Loss**: Handles variable-length sequences without alignment
- **Comprehensive Logging**: Training history and test results
- **Easy Usage**: Batch scripts for quick training/testing

## 📈 Usage Examples

### Custom Image Testing
```python
from src.models.thai_crnn import load_model, load_char_mapping
from src.testing.test_thai_crnn_clean import preprocess_image, ctc_greedy_decode

# Load model
char_to_idx = load_char_mapping()
model = load_model("models/thai_crnn_ctc_best.pth", char_to_idx)

# Test custom image
image = preprocess_image("path/to/your/image.jpg")
output = model(image)
predicted_text = ctc_greedy_decode(output, char_to_idx)
```

## 🔧 Configuration

Edit `configs/model_config.json` to customize:
- Model parameters
- Training settings
- Data paths
- GPU settings

## 📝 Logs and Results

- Training history: `logs/training_history.json`
- Test results: `logs/test_results.json`
- Model checkpoints: `models/`

## 🎮 RTX 5090 Features

- Memory optimization for 24GB VRAM
- PyTorch nightly build compatibility
- sm_120 architecture support
- Optimized batch processing

## 🔄 Model Improvement Tips

1. **Reduce character classes** from 881 to 50-100 most common
2. **Increase training epochs** to 50-100
3. **Data augmentation** for better generalization
4. **Learning rate scheduling** for fine-tuning

## 📦 Dependencies

- PyTorch 2.9+ (nightly build for RTX 5090)
- torchvision
- Pillow
- NumPy

## 🏆 Performance

Current model status:
- ✅ RTX 5090 training: Working
- ✅ CTC loss convergence: Working  
- ⚠️ Model accuracy: Needs improvement (currently overfitting)

## 🤝 Contributing

This is a clean, organized codebase ready for:
- Model architecture improvements
- Training optimization
- Dataset expansion
- Performance enhancements
