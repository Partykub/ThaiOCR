#!/usr/bin/env python3
"""
Thai CRNN Training with Real Dataset - RTX 5090 Optimized
=========================================================

Train CRNN model using Thai OCR dataset (5000 samples)
Using th_dict.txt character mapping (882 characters)
Optimized for RTX 5090 with PyTorch nightly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import warnings
import json
import time
import random
from pathlib import Path
import numpy as np

# RTX 5090 Compatibility Settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
warnings.filterwarnings('ignore', message='.*sm_120.*')

class ThaiOCRDataset(Dataset):
    """Thai OCR Dataset for CRNN Training"""
    
    def __init__(self, dataset_dir, char_to_idx, split='train', train_ratio=0.8):
        self.dataset_dir = Path(dataset_dir)
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load samples
        self.samples = self._load_samples()
        
        # Split train/validation
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"📊 {split.upper()} Dataset: {len(self.samples)} samples")
    
    def _load_samples(self):
        """Load image-text pairs from labels.txt"""
        labels_file = self.dataset_dir / "labels.txt"
        samples = []
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    img_name, text = parts[0], parts[1]
                    img_path = self.dataset_dir / "images" / img_name
                    
                    if img_path.exists():
                        samples.append((str(img_path), text))
        
        print(f"📈 Loaded {len(samples)} valid samples")
        return samples
    
    def text_to_sequence(self, text):
        """Convert text to sequence of character indices"""
        sequence = []
        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                # Use <UNK> token if available, otherwise skip
                if '<UNK>' in self.char_to_idx:
                    sequence.append(self.char_to_idx['<UNK>'])
        return sequence
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros(3, 64, 128)
        
        # Convert text to sequence
        sequence = self.text_to_sequence(text)
        
        return image, torch.tensor(sequence, dtype=torch.long), text

class ThaiCRNN(nn.Module):
    """CRNN Model for Thai OCR"""
    
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
            nn.MaxPool2d((2, 1)),  # 8x32 (keep width)
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 4x32
        )
        
        # Calculate feature map dimensions
        self.feature_height = 4
        self.feature_width = 32
        
        # RNN Sequence Modeling
        rnn_input_size = 512 * self.feature_height
        self.rnn = nn.LSTM(rnn_input_size, hidden_size, 
                          num_layers=2, bidirectional=True, 
                          batch_first=True, dropout=0.1)
        
        # Classification
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)  # [B, 512, 4, 32]
        
        # Reshape for RNN: [B, sequence_length, feature_size]
        features = features.permute(0, 3, 1, 2)  # [B, 32, 512, 4]
        features = features.contiguous().view(batch_size, self.feature_width, -1)
        
        # RNN processing
        rnn_out, _ = self.rnn(features)  # [B, 32, hidden_size*2]
        
        # Classification
        output = self.classifier(rnn_out)  # [B, 32, num_classes]
        
        return output

def load_char_mapping():
    """Load character mapping from th_dict.txt"""
    dict_file = Path("thai-letters/th_dict.txt")
    
    char_to_idx = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char:  # Skip empty lines
                char_to_idx[char] = idx
    
    # Add special tokens
    char_to_idx['<BLANK>'] = len(char_to_idx)
    char_to_idx['<UNK>'] = len(char_to_idx)
    
    print(f"📝 Character mapping loaded: {len(char_to_idx)} characters")
    return char_to_idx

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    images, sequences, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Find max sequence length in this batch
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to same length and create masks
    padded_sequences = []
    sequence_lengths = []
    
    for seq in sequences:
        length = len(seq)
        # Pad with <BLANK> token index
        padded = torch.full((max_len,), fill_value=880, dtype=torch.long)  # 880 is <BLANK> index
        padded[:length] = seq
        padded_sequences.append(padded)
        sequence_lengths.append(length)
    
    sequences = torch.stack(padded_sequences)
    sequence_lengths = torch.tensor(sequence_lengths)
    
    return images, sequences, sequence_lengths, texts

def verify_rtx5090():
    """Verify RTX 5090 is ready for training"""
    print("🔍 RTX 5090 Training Verification")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CRITICAL: NO GPU DETECTED - TRAINING ABORTED")
    
    device_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU Device: {device_name}")
    print(f"✅ GPU Memory: {memory_gb:.1f} GB")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    
    if "RTX 5090" in device_name:
        print("🚀 RTX 5090 DETECTED - READY FOR TRAINING!")
    
    return True

def train_model():
    """Train Thai CRNN model on real dataset"""
    print("🏋️ Thai CRNN Training Started")
    print("=" * 50)
    
    # Verify GPU
    verify_rtx5090()
    device = torch.device('cuda')
    
    # Load character mapping
    char_to_idx = load_char_mapping()
    num_classes = len(char_to_idx)
    
    # Create datasets
    dataset_dir = "thai-letters/thai_ocr_dataset"
    
    train_dataset = ThaiOCRDataset(dataset_dir, char_to_idx, split='train')
    val_dataset = ThaiOCRDataset(dataset_dir, char_to_idx, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2)
    
    # Create model
    model = ThaiCRNN(num_classes=num_classes).to(device)
    print(f"✅ Model created: {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<BLANK>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    
    print(f"🔄 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (images, sequences, lengths, texts) in enumerate(train_loader):
            images = images.to(device)
            sequences = sequences.to(device)
            
            # Forward pass
            outputs = model(images)  # [B, seq_len, num_classes]
            
            # Calculate loss - handle sequence length mismatch
            B, S_out, C = outputs.shape  # Model output sequence length
            B, S_tgt = sequences.shape   # Target sequence length
            
            # Adjust sequences to match model output length
            if S_out != S_tgt:
                if S_out > S_tgt:
                    # Pad target sequences
                    pad_size = S_out - S_tgt
                    padding = torch.full((B, pad_size), fill_value=char_to_idx['<BLANK>'], 
                                       dtype=sequences.dtype, device=sequences.device)
                    sequences = torch.cat([sequences, padding], dim=1)
                else:
                    # Truncate target sequences
                    sequences = sequences[:, :S_out]
            
            # Now calculate loss
            outputs_flat = outputs.view(-1, C)  # [B*S, C]
            sequences_flat = sequences.view(-1)  # [B*S]
            
            loss = criterion(outputs_flat, sequences_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += len(images)
            
            if batch_idx % 50 == 0:
                memory_used = torch.cuda.memory_allocated() / 1024**2
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, GPU={memory_used:.1f}MB")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for images, sequences, lengths, texts in val_loader:
                images = images.to(device)
                sequences = sequences.to(device)
                
                # Calculate loss - handle sequence length mismatch  
                B, S_out, C = outputs.shape  # Model output sequence length
                B, S_tgt = sequences.shape   # Target sequence length
                
                # Adjust sequences to match model output length
                if S_out != S_tgt:
                    if S_out > S_tgt:
                        # Pad target sequences
                        pad_size = S_out - S_tgt
                        padding = torch.full((B, pad_size), fill_value=char_to_idx['<BLANK>'], 
                                           dtype=sequences.dtype, device=sequences.device)
                        sequences = torch.cat([sequences, padding], dim=1)
                    else:
                        # Truncate target sequences
                        sequences = sequences[:, :S_out]
                
                # Now calculate loss
                outputs_flat = outputs.view(-1, C)  # [B*S, C]
                sequences_flat = sequences.view(-1)  # [B*S]
                
                loss = criterion(outputs_flat, sequences_flat)
                
                val_loss += loss.item()
                val_samples += len(images)
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch {epoch+1}/{num_epochs}:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'char_to_idx': char_to_idx
            }, 'build-model-th/thai_crnn_best.pth')
            print(f"💾 Best model saved (val_loss: {avg_val_loss:.4f})")
        
        scheduler.step()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print("🎉 Training completed!")
    return model, char_to_idx

if __name__ == "__main__":
    try:
        model, char_to_idx = train_model()
        print("✅ Thai CRNN training successful!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
