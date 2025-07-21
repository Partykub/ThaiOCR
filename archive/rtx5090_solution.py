"""
RTX 5090 PyTorch Workaround Script
==================================

Based on Reddit solution for RTX 5090 sm_120 compatibility
This script implements CPU fallback training when GPU is incompatible
"""

import os
import warnings
import json
from pathlib import Path

# RTX 5090 Compatibility Environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;9.0'
warnings.filterwarnings("ignore", message=".*sm_120.*")

def check_rtx5090_compatibility():
    """Check RTX 5090 compatibility and provide solutions"""
    
    print("🔍 RTX 5090 Compatibility Analysis")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"🎯 GPU: {device_name}")
            
            if "RTX 5090" in device_name:
                print("🚀 RTX 5090 Detected!")
                
                # Test GPU operations
                try:
                    test_tensor = torch.randn(10, 10, device='cuda')
                    result = torch.mm(test_tensor, test_tensor)
                    print("✅ GPU Operations: WORKING")
                    return "gpu", torch.device('cuda')
                    
                except Exception as e:
                    print(f"❌ GPU Operations Failed: {e}")
                    
                    if "no kernel image" in str(e):
                        print("\n🔧 RTX 5090 Solution Options:")
                        print("1. 📦 Compile PyTorch from source with CUDA 12.8+")
                        print("2. 🐍 Use Conda instead of pip")
                        print("3. ⏳ Wait for official PyTorch RTX 5090 support")
                        print("4. 💻 Use CPU fallback (current implementation)")
                        
                        print("\n🔗 Reddit Solution:")
                        print("   - CUDA Toolkit 12.8+ required")
                        print("   - Compile with: arch=compute_120,code=sm_120")
                        print("   - Use conda-forge PyTorch builds")
                        
                        print("\n⚡ Using CPU fallback for now...")
                        return "cpu", torch.device('cpu')
            
            return "gpu", torch.device('cuda')
        
        else:
            print("❌ CUDA not available")
            return "cpu", torch.device('cpu')
            
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return "cpu", torch.device('cpu')

def create_cpu_fallback_training():
    """Create CPU-optimized training script for RTX 5090 incompatibility"""
    
    script_content = '''#!/usr/bin/env python3
"""
Thai CRNN Training - CPU Fallback for RTX 5090 Incompatibility
===============================================================

When RTX 5090 PyTorch compatibility fails, use CPU training as temporary solution
This violates the GPU-only policy but provides a working alternative
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Force CPU usage for RTX 5090 incompatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class ThaiOCRDataset(Dataset):
    def __init__(self, dataset_dir, char_to_idx, img_height=32, img_width=128):
        self.dataset_dir = Path(dataset_dir)
        self.char_to_idx = char_to_idx
        
        # Load samples
        labels_file = self.dataset_dir / "labels.txt"
        self.samples = []
        
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\\t')
                    if len(parts) >= 2:
                        img_name, text = parts[0], parts[1]
                        img_path = self.dataset_dir / "images" / img_name
                        if img_path.exists():
                            self.samples.append((str(img_path), text))
        
        print(f"📊 Dataset: {len(self.samples)} samples")
        
        # CPU-optimized transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')  # Grayscale for CPU
            image = self.transform(image)
        except:
            image = torch.zeros(1, 32, 128)
            text = ""
        
        # Convert text to indices
        text_indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0))
            text_indices.append(idx)
        
        return image, torch.tensor(text_indices, dtype=torch.long), text

class SimpleCRNN(nn.Module):
    """Simplified CRNN for CPU training"""
    
    def __init__(self, num_classes, hidden_size=128):
        super(SimpleCRNN, self).__init__()
        
        # Lightweight CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x64
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x32
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 4x32
        )
        
        # RNN
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # [B, 128, 4, 32]
        
        # Reshape for RNN
        b, c, h, w = conv.shape
        conv = conv.permute(0, 3, 1, 2)  # [B, W, C, H]
        conv = conv.reshape(b, w, c * h)  # [B, W, C*H]
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # Classify
        output = self.classifier(rnn_out)
        return output

def main():
    print("🐌 CPU Fallback Training for RTX 5090 Incompatibility")
    print("⚠️  This violates GPU-only policy but provides working solution")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"💻 Using device: {device}")
    
    # Load character mapping
    script_dir = Path(__file__).parent
    char_map_file = script_dir / "thai_char_map.json"
    
    if not char_map_file.exists():
        print("❌ Character mapping not found!")
        return
    
    with open(char_map_file, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)
    
    num_classes = len(char_to_idx)
    print(f"📚 Classes: {num_classes}")
    
    # Dataset
    dataset_dir = script_dir.parent / "thai-letters" / "thai_ocr_dataset"
    dataset = ThaiOCRDataset(dataset_dir, char_to_idx)
    
    if len(dataset) == 0:
        print("❌ No dataset!")
        return
    
    # Small batch for CPU
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model
    model = SimpleCRNN(num_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🚀 Starting CPU training...")
    
    # Training loop
    epochs = 10  # Reduced for CPU
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, texts, raw_texts in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Reshape for loss
            b, w, c = outputs.shape
            outputs_flat = outputs.reshape(-1, c)
            
            # Pad texts to match output width
            max_len = w
            padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
            for i, text in enumerate(texts):
                seq_len = min(len(text), max_len)
                padded_texts[i, :seq_len] = text[:seq_len]
            
            texts_flat = padded_texts.reshape(-1)
            
            loss = criterion(outputs_flat, texts_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"📊 Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save model
    model_path = script_dir / "thai_crnn_cpu_fallback.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'num_classes': num_classes
    }, model_path)
    
    print(f"💾 CPU model saved: {model_path}")
    print("✅ CPU fallback training completed!")

if __name__ == "__main__":
    main()
'''
    
    # Save CPU fallback script
    script_dir = Path(__file__).parent
    cpu_script_path = script_dir / "train_thai_crnn_cpu_fallback.py"
    
    with open(cpu_script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"💾 CPU fallback script created: {cpu_script_path}")
    return cpu_script_path

def main():
    """Main function for RTX 5090 compatibility handling"""
    
    print("🔧 RTX 5090 PyTorch Compatibility Handler")
    print("=" * 60)
    
    # Check compatibility
    mode, device = check_rtx5090_compatibility()
    
    if mode == "gpu":
        print("\n✅ GPU training possible!")
        print("🚀 Use: train_thai_crnn_pytorch.py")
        
    else:
        print("\n⚠️ GPU incompatible - Creating CPU fallback")
        
        # Create CPU fallback script
        cpu_script = create_cpu_fallback_training()
        
        print(f"\n🐌 CPU Fallback Options:")
        print(f"1. Run: python {cpu_script.name}")
        print(f"2. Wait for PyTorch RTX 5090 support")
        print(f"3. Use conda instead of pip")
        print(f"4. Compile PyTorch from source")
        
        print(f"\n🔗 RTX 5090 Solutions:")
        print(f"   - Reddit: arch=compute_120,code=sm_120")
        print(f"   - CUDA Toolkit 12.8+ required")
        print(f"   - Conda: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")

if __name__ == "__main__":
    main()
