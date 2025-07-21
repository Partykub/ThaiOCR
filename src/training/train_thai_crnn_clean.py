#!/usr/bin/env python3
"""
Thai CRNN Training Script - Clean Version
==========================================

Train CRNN model using Thai OCR dataset with CTC Loss
Optimized for RTX 5090 GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import warnings
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.thai_crnn import ThaiCRNN, load_char_mapping, create_model
from utils.dataset import ThaiOCRDataset, collate_fn

# RTX 5090 Compatibility Settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
warnings.filterwarnings('ignore', message='.*sm_120.*')


def collate_fn_ctc(batch):
    """Custom collate function for CTC training"""
    images, sequences, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Get sequence lengths and concatenate for CTC
    sequence_lengths = torch.tensor([len(seq) for seq in sequences])
    concatenated_sequences = torch.cat(sequences)
    
    return images, concatenated_sequences, sequence_lengths, texts


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
    """Train Thai CRNN model with CTC Loss"""
    print("🏋️ Thai CRNN Training with CTC Loss")
    print("=" * 50)
    
    # Verify GPU
    verify_rtx5090()
    device = torch.device('cuda')
    
    # Load character mapping
    char_to_idx = load_char_mapping()
    num_classes = len(char_to_idx)
    blank_idx = char_to_idx['<BLANK>']
    
    print(f"📚 Character classes: {num_classes} (including blank)")
    print(f"🎯 Blank token index: {blank_idx}")
    
    # Create datasets
    dataset_dir = "thai-letters/thai_ocr_dataset"
    
    train_dataset = ThaiOCRDataset(dataset_dir, char_to_idx, split='train')
    val_dataset = ThaiOCRDataset(dataset_dir, char_to_idx, split='val')
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                             collate_fn=collate_fn_ctc, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                           collate_fn=collate_fn_ctc, num_workers=0)
    
    # Create model
    model = create_model(char_to_idx)
    model.to(device)
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Training parameters
    num_epochs = 20
    best_val_loss = float('inf')
    
    # Training loop
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (images, targets, target_lengths, texts) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            log_probs = model(images)
            
            # CTC loss expects log probabilities
            log_probs = torch.log_softmax(log_probs, dim=2)
            
            # Input lengths (sequence length from model output)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            
            # Calculate loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                log_probs = model(images)
                log_probs = torch.log_softmax(log_probs, dim=2)
                
                input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        epoch_time = time.time() - start_time
        
        print(f"  ⏱️  Time: {epoch_time:.1f}s")
        print(f"  📈 Train Loss: {avg_train_loss:.4f}")
        print(f"  📉 Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  🎯 Learning Rate: {current_lr:.6f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  💾 New best model! Saving checkpoint...")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'char_to_idx': char_to_idx
            }, 'models/thai_crnn_ctc_best.pth')
    
    # Save training history
    with open('logs/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 Training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print("💾 Model saved to: models/thai_crnn_ctc_best.pth")
    print("📈 History saved to: logs/training_history.json")


if __name__ == "__main__":
    train_model()
