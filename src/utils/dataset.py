#!/usr/bin/env python3
"""
Thai OCR Dataset Utilities
==========================

Dataset classes and utilities for Thai CRNN training
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from pathlib import Path


class ThaiOCRDataset(Dataset):
    """Thai OCR Dataset for CRNN Training with CTC"""
    
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
    
    def _load_samples(self):
        """Load image-text pairs from dataset"""
        samples = []
        labels_file = self.dataset_dir / "labels.txt"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        text = parts[1]
                        img_path = self.dataset_dir / "images" / img_name
                        
                        if img_path.exists():
                            samples.append((str(img_path), text))
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def text_to_sequence(self, text):
        """Convert text to sequence of character indices (excluding blank)"""
        sequence = []
        for char in text:
            if char in self.char_to_idx:
                # Skip blank token for CTC
                idx = self.char_to_idx[char]
                if idx != self.char_to_idx['<BLANK>']:
                    sequence.append(idx)
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
            image = torch.zeros(3, 64, 128)
        
        # Convert text to sequence (without blank tokens)
        sequence = self.text_to_sequence(text)
        
        return image, torch.tensor(sequence, dtype=torch.long), text


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    images, sequences, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Get sequence lengths
    seq_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad sequences
    max_len = max(seq_lengths)
    padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return images, padded_sequences, seq_lengths, texts
