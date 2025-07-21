# Training Guide for PaddleOCR Thai Language Model

## Overview
This guide walks you through training a custom Thai language recognition model using PaddleOCR and the existing Thai dataset.

## Prerequisites
- Completed [Installation Guide](installation_guide.md)
- PaddlePaddle GPU with RTX 5090 support
- Thai dataset (available in `thai-letters/thai_ocr_dataset/`)

## Dataset Preparation

### 1. Verify Existing Dataset
```bash
# Check dataset structure
cd c:\Users\admin\Documents\paddlepadle\thai-letters\thai_ocr_dataset

# Should contain:
# - images/     (training images)
# - labels.txt  (image-text pairs)
```

### 2. Prepare Thai Character Dictionary
```bash
# Use existing thai dictionary
cp th_dict.txt ../paddleocr/ppocr/utils/dict/thai_dict.txt

# Or create custom dictionary
python -c "
import codecs
with codecs.open('th_dict.txt', 'r', 'utf-8') as f:
    chars = f.read().strip()
    print(f'Thai characters: {len(chars)} unique chars')
    print(f'Sample: {chars[:20]}...')
"
```

### 3. Convert Dataset Format
Create dataset conversion script:

```python
# dataset_converter.py
import os
import json
from PIL import Image

def convert_to_paddleocr_format():
    """Convert existing dataset to PaddleOCR format"""
    
    # Read existing labels
    with open('thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    train_data = []
    val_data = []
    
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            image_path = f"images/{parts[0]}"
            text = parts[1]
            
            # Split 80% train, 20% validation
            if i % 5 == 0:
                val_data.append(f"{image_path}\t{text}\n")
            else:
                train_data.append(f"{image_path}\t{text}\n")
    
    # Write train/val splits
    with open('train_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open('val_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")

if __name__ == "__main__":
    convert_to_paddleocr_format()
```

## Configuration Setup

### 1. Create Thai Recognition Config
Create `configs/rec/thai_svtr_tiny.yml`:

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_thai_svtr_tiny
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: ./pretrain_models/rec_mv3_none_bilstm_ctc_v2.0_train
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: ./ppocr/utils/dict/thai_dict.txt
  character_type: thai
  max_text_length: 25
  infer_mode: false
  use_space_char: true
  distributed: true
  save_res_path: ./output/rec/predicts_thai.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: SVTRNet
    img_size: [64, 256]
    out_char_num: 25
    out_channels: 192
    patch_merging: 'Conv'
    arch: 'tiny'
    last_stage: true
    prenorm: false
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: true
  Head:
    name: CTCHead
    mid_channels: 96
    fc_decay: 0.00002

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./thai-letters/thai_ocr_dataset/
    ext_op_transform_idx: 1
    label_file_list:
    - ./thai-letters/thai_ocr_dataset/train_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode:
    - RecResizeImg:
        image_shape: [3, 64, 256]
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: true
    batch_size_per_card: 128
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./thai-letters/thai_ocr_dataset/
    label_file_list:
    - ./thai-letters/thai_ocr_dataset/val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode:
    - RecResizeImg:
        image_shape: [3, 64, 256]
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 2
```

### 2. Download Pretrained Model
```bash
# Download pretrained English model as starting point
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
cd ./pretrain_models/ && tar xf ch_PP-OCRv3_rec_train.tar && cd ../
```

## Training Process

### 1. Start Training
```bash
# Single GPU training
python tools/train.py -c configs/rec/thai_svtr_tiny.yml

# Multi-GPU training (if available)
python -m paddle.distributed.launch --gpus="0,1" tools/train.py -c configs/rec/thai_svtr_tiny.yml
```

### 2. Monitor Training Progress
```bash
# View training logs
tail -f train.log

# Expected output:
# [2025/07/21 10:30:15] epoch: [1/100], global_step: 10, lr: 0.001000, loss: 3.2456, acc: 0.1234, norm: 1.0, time: 0.123
# [2025/07/21 10:30:30] epoch: [1/100], global_step: 20, lr: 0.001000, loss: 2.9876, acc: 0.2345, norm: 1.0, time: 0.115
```

### 3. Evaluate Model
```bash
# Evaluate during training
python tools/eval.py -c configs/rec/thai_svtr_tiny.yml -o Global.checkpoints=./output/rec_thai_svtr_tiny/best_accuracy

# Manual evaluation
python tools/eval.py -c configs/rec/thai_svtr_tiny.yml -o Global.checkpoints=./output/rec_thai_svtr_tiny/latest
```

## Model Export and Testing

### 1. Export Inference Model
```bash
# Convert trained model to inference format
python tools/export_model.py -c configs/rec/thai_svtr_tiny.yml -o Global.pretrained_model=./output/rec_thai_svtr_tiny/best_accuracy Global.save_inference_dir=./inference/thai_rec/
```

### 2. Test Inference
```python
# test_thai_inference.py
import cv2
import numpy as np
from paddleocr import PaddleOCR

def test_thai_ocr():
    """Test Thai OCR with custom model"""
    
    # Initialize with custom Thai model
    ocr = PaddleOCR(
        use_angle_cls=True,
        rec_model_dir='./inference/thai_rec/',
        rec_char_dict_path='./ppocr/utils/dict/thai_dict.txt',
        use_gpu=True,
        show_log=False
    )
    
    # Test with validation images
    test_images = [
        'thai-letters/thai_ocr_dataset/images/000001.jpg',
        'thai-letters/thai_ocr_dataset/images/000002.jpg',
        'thai-letters/thai_ocr_dataset/images/000003.jpg'
    ]
    
    for img_path in test_images:
        result = ocr.ocr(img_path, cls=True)
        print(f"Image: {img_path}")
        print(f"Result: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_thai_ocr()
```

## Performance Optimization

### 1. Training Hyperparameters
```yaml
# For better accuracy
Optimizer:
  lr:
    learning_rate: 0.0005  # Lower learning rate
    warmup_epoch: 5        # More warmup

# For faster training
Train:
  loader:
    batch_size_per_card: 256  # Larger batch size if GPU memory allows
```

### 2. Data Augmentation
```yaml
Train:
  dataset:
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode:
    - RecAug:              # Add data augmentation
    - RecResizeImg:
        image_shape: [3, 64, 256]
    - KeepKeys:
        keep_keys: [image, label, length]
```

### 3. Mixed Precision Training
```bash
# Enable mixed precision for faster training
export FLAGS_conv_workspace_size_limit=512
export FLAGS_max_inplace_grad_add=5
python tools/train.py -c configs/rec/thai_svtr_tiny.yml --amp
```

## Comparison with Existing CRNN Model

### 1. Benchmark Test
```python
# benchmark_comparison.py
import time
import cv2
from paddleocr import PaddleOCR
import sys
sys.path.append('thai-license-plate-recognition-CRNN')

# Import existing CRNN model
from Model import load_model
from parameter import letters, img_w, img_h

def benchmark_models():
    """Compare PaddleOCR vs existing CRNN"""
    
    # Load models
    paddle_ocr = PaddleOCR(rec_model_dir='./inference/thai_rec/')
    crnn_model = load_model('thai-license-plate-recognition-CRNN/Model_LSTM+BN5--thai-v3.h5')
    
    test_images = ['test_image_1.jpg', 'test_image_2.jpg']
    
    for img_path in test_images:
        # PaddleOCR timing
        start_time = time.time()
        paddle_result = paddle_ocr.ocr(img_path)
        paddle_time = time.time() - start_time
        
        # CRNN timing (simplified)
        start_time = time.time()
        # crnn_result = predict_crnn(img_path, crnn_model)
        crnn_time = time.time() - start_time
        
        print(f"Image: {img_path}")
        print(f"PaddleOCR: {paddle_result} ({paddle_time:.3f}s)")
        # print(f"CRNN: {crnn_result} ({crnn_time:.3f}s)")
        print("-" * 50)

if __name__ == "__main__":
    benchmark_models()
```

## Troubleshooting Training Issues

### Common Problems

#### 1. Out of Memory
```bash
# Reduce batch size
batch_size_per_card: 64  # instead of 128

# Or use gradient accumulation
export FLAGS_max_inplace_grad_add=8
```

#### 2. Low Accuracy
```bash
# Check data quality
python tools/infer_rec.py -c configs/rec/thai_svtr_tiny.yml -o Global.pretrained_model=./output/rec_thai_svtr_tiny/latest Global.infer_img=./thai-letters/thai_ocr_dataset/images/000001.jpg

# Verify character dictionary
python -c "
with open('ppocr/utils/dict/thai_dict.txt', 'r', encoding='utf-8') as f:
    chars = f.read()
    print(f'Dictionary size: {len(chars)}')
    print(f'Characters: {chars}')
"
```

#### 3. Training Stops
```bash
# Resume from checkpoint
python tools/train.py -c configs/rec/thai_svtr_tiny.yml -o Global.checkpoints=./output/rec_thai_svtr_tiny/latest
```

## Next Steps

After successful training:
1. [API Reference](api_reference.md) - Integrate model into applications
2. [Deployment Guide](deployment_guide.md) - Deploy to production
3. Fine-tune for specific use cases (license plates, documents, etc.)
