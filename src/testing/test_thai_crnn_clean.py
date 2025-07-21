#!/usr/bin/env python3
"""
Thai CRNN Model Testing Script - Clean Version
===============================================

Test trained Thai CRNN model with CTC decoding
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.thai_crnn import load_model, load_char_mapping


def ctc_greedy_decode(log_probs, char_to_idx):
    """CTC Greedy decoding"""
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    blank_idx = char_to_idx['<BLANK>']
    
    # Get the most probable character at each time step
    _, predicted_indices = torch.max(log_probs, dim=2)
    predicted_indices = predicted_indices.squeeze(1)  # Remove batch dimension
    
    # Remove consecutive duplicates and blanks
    decoded_chars = []
    prev_idx = None
    
    for idx in predicted_indices:
        idx = idx.item()
        if idx != prev_idx and idx != blank_idx:
            if idx in idx_to_char:
                decoded_chars.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(decoded_chars)


def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def test_model():
    """Test the trained Thai CRNN model"""
    print("🧪 Thai CRNN Model Testing")
    print("=" * 40)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load character mapping and model
    char_to_idx = load_char_mapping()
    model_path = "models/thai_crnn_ctc_best.pth"
    
    try:
        model = load_model(model_path, char_to_idx, device)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Test images
    test_images = [
        "thai-letters/thai_ocr_dataset/images/000001.jpg",
        "thai-letters/thai_ocr_dataset/images/000010.jpg",
        "thai-letters/thai_ocr_dataset/images/000020.jpg",
        "thai-letters/thai_ocr_dataset/images/000030.jpg",
        "thai-letters/thai_ocr_dataset/images/000040.jpg",
    ]
    
    results = []
    
    print("\n🔍 Testing Results:")
    print("-" * 40)
    
    for i, img_path in enumerate(test_images, 1):
        if not Path(img_path).exists():
            print(f"⚠️  Image {i}: File not found - {img_path}")
            continue
        
        try:
            # Preprocess image
            image = preprocess_image(img_path)
            image = image.to(device)
            
            # Model inference
            with torch.no_grad():
                output = model(image)
                log_probs = torch.log_softmax(output, dim=2)
                
                # Decode prediction
                predicted_text = ctc_greedy_decode(log_probs, char_to_idx)
                
                # Get confidence (average probability of predicted sequence)
                max_probs = torch.max(torch.softmax(output, dim=2), dim=2)[0]
                confidence = torch.mean(max_probs).item()
            
            # Store result
            result = {
                'image': Path(img_path).name,
                'predicted_text': predicted_text,
                'confidence': confidence
            }
            results.append(result)
            
            print(f"📷 Image {i}: {Path(img_path).name}")
            print(f"   🔤 Predicted: '{predicted_text}'")
            print(f"   📊 Confidence: {confidence:.4f}")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    # Save results
    results_path = "logs/test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Summary
    print("\n📊 Testing Summary:")
    print("-" * 20)
    print(f"Total images tested: {len(results)}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Average confidence: {avg_confidence:.4f}")
        
        non_empty_predictions = [r for r in results if r['predicted_text']]
        print(f"Non-empty predictions: {len(non_empty_predictions)}/{len(results)}")


if __name__ == "__main__":
    test_model()
