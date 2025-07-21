#!/usr/bin/env python3
"""
Thai CRNN Model Testing Script - RTX 5090 Optimized
===================================================

Test trained Thai CRNN model with real images
Load best model and perform Thai text recognition
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import warnings
import json
from pathlib import Path
import numpy as np

# RTX 5090 Compatibility Settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
warnings.filterwarnings('ignore', message='.*sm_120.*')

class ThaiCRNN(nn.Module):
    """CRNN Model for Thai OCR with CTC (same as training)"""
    
    def __init__(self, num_classes, img_height=64, img_width=128, hidden_size=256):
        super(ThaiCRNN, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x32
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 8x32
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 4x32
        )
        
        # RNN Sequence Modeling
        rnn_input_size = 512 * 4  # 4 is the height after CNN
        self.rnn = nn.LSTM(rnn_input_size, hidden_size, 
                          num_layers=2, bidirectional=True, 
                          batch_first=True, dropout=0.1)
        
        # Classification (including blank token for CTC)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)  # [B, 512, 4, 32]
        
        # Reshape for RNN: [B, sequence_length, feature_size]
        features = features.permute(0, 3, 1, 2)  # [B, 32, 512, 4]
        features = features.contiguous().view(batch_size, 32, -1)
        
        # RNN processing
        rnn_out, _ = self.rnn(features)  # [B, 32, hidden_size*2]
        
        # Classification
        output = self.classifier(rnn_out)  # [B, 32, num_classes]
        
        return output

def load_trained_model(model_path, device):
    """Load the trained Thai CRNN model"""
    print(f"📂 Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # Create model
    num_classes = len(char_to_idx)
    model = ThaiCRNN(num_classes=num_classes).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Classes: {num_classes}")
    print(f"📈 Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"🏆 Trained epoch: {checkpoint['epoch']}")
    
    return model, char_to_idx, idx_to_char

def preprocess_image(image_path):
    """Preprocess image for CRNN input"""
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None, None

def ctc_decode(predictions, idx_to_char, blank_idx):
    """Simple CTC decoding without language model"""
    # Get the best path (greedy decoding)
    best_path = torch.argmax(predictions, dim=-1)  # [T]
    
    # Remove consecutive duplicates and blanks
    decoded = []
    prev_idx = None
    
    for idx in best_path:
        idx = idx.item()
        if idx != blank_idx and idx != prev_idx:
            if idx in idx_to_char:
                decoded.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(decoded)

def predict_text(model, image_tensor, char_to_idx, idx_to_char, device):
    """Predict text from image using trained CRNN"""
    blank_idx = char_to_idx['<BLANK>']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)  # [B, T, C]
        
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=-1)
        
        # Get predictions for first (and only) image in batch
        predictions = probs[0]  # [T, C]
        
        # Decode using CTC
        predicted_text = ctc_decode(predictions, idx_to_char, blank_idx)
        
        # Get confidence score (average of max probabilities)
        confidence = torch.max(probs[0], dim=-1)[0].mean().item()
        
        return predicted_text, confidence

def test_model():
    """Test the trained Thai CRNN model"""
    print("🧪 Thai CRNN Model Testing")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU")
    
    # Load trained model
    model_path = "build-model-th/thai_crnn_ctc_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run training first!")
        return
    
    model, char_to_idx, idx_to_char = load_trained_model(model_path, device)
    
    # Test with sample images from dataset
    dataset_dir = Path("thai-letters/thai_ocr_dataset")
    images_dir = dataset_dir / "images"
    labels_file = dataset_dir / "labels.txt"
    
    # Load some test samples
    test_samples = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Take first 10 samples for testing
        for line in lines[:10]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name, true_text = parts[0], parts[1]
                img_path = images_dir / img_name
                if img_path.exists():
                    test_samples.append((str(img_path), true_text))
    
    print(f"🔍 Testing with {len(test_samples)} sample images...")
    print("=" * 50)
    
    # Test each sample
    correct_predictions = 0
    total_predictions = len(test_samples)
    
    for i, (img_path, true_text) in enumerate(test_samples):
        print(f"\n📷 Test {i+1}/{total_predictions}")
        print(f"Image: {Path(img_path).name}")
        print(f"True text: '{true_text}'")
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(img_path)
        
        if image_tensor is None:
            print("❌ Failed to load image")
            continue
        
        # Predict
        predicted_text, confidence = predict_text(model, image_tensor, 
                                                char_to_idx, idx_to_char, device)
        
        print(f"Predicted: '{predicted_text}'")
        print(f"Confidence: {confidence:.4f}")
        
        # Check accuracy
        if predicted_text.strip() == true_text.strip():
            print("✅ CORRECT!")
            correct_predictions += 1
        else:
            print("❌ INCORRECT")
        
        print("-" * 30)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    print(f"\n📊 Testing Results:")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if accuracy > 70:
        print("🎉 Great! Model is performing well!")
    elif accuracy > 40:
        print("👍 Good! Model shows promising results!")
    else:
        print("🔧 Model needs more training or parameter tuning")

def test_single_image(image_path):
    """Test model with a single image"""
    print(f"🧪 Testing single image: {image_path}")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = "build-model-th/thai_crnn_ctc_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    model, char_to_idx, idx_to_char = load_trained_model(model_path, device)
    
    # Preprocess and predict
    image_tensor, original_image = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    predicted_text, confidence = predict_text(model, image_tensor, 
                                            char_to_idx, idx_to_char, device)
    
    print(f"📝 Predicted text: '{predicted_text}'")
    print(f"🎯 Confidence: {confidence:.4f}")
    
    return predicted_text, confidence

if __name__ == "__main__":
    try:
        # Test with dataset samples
        test_model()
        
        # Example: Test with a specific image
        # test_single_image("thai-letters/thai_ocr_dataset/images/000001.jpg")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        raise
